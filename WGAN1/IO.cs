using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace WGAN1
{
    static class IO
    {
        public static bool Testing = false;
        public static bool LabelReaderRunning = false;
        public static bool ImageReaderRunning = false;
        static readonly string GWBPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\GeneratorWBs.txt";
        static readonly string DWBPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\DiscriminatorWBs.txt";
        static readonly string TrainImagePath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\train-images.idx3-ubyte";
        static readonly string TrainLabelPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\train-labels.idx1-ubyte";
        static readonly string TestLabelPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\test-labels.idx1-ubyte";
        static readonly string TestImagePath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\test-images.idx3-ubyte";
        private static string LabelPath = Testing ? TestLabelPath : TrainLabelPath;
        private static string ImagePath = Testing ? TestImagePath : TrainImagePath;
        static int LabelOffset = 8;
        static int ImageOffset = 16;
        static int Resolution = 28;

        public static double[] FindNextNumber(int number)
        {
            //Find the next number
            while(ReadNextLabel() != number)
            { 
                ImageOffset++; 
            }
            //Return the image found at the now found index
            return ReadNextImage();
        }
       
        //Simple code to read a single number from a file, offset by a byte of metadata
        static int ReadNextLabel()
        {
            //Singleton process
            if (LabelReaderRunning) { throw new Exception("Already accessing file"); }

            FileStream fs = File.OpenRead(LabelPath);
            //Reset parameters and decrement NN hyperparameters upon new epoch (currently disabled)
            if (!(LabelOffset < fs.Length)) { LabelOffset = 8; ImageOffset = 16; }

            fs.Position = LabelOffset;
            byte[] b = new byte[1];
            try
            {
                fs.Read(b, 0, 1);
            }
            catch (Exception ex) { Console.WriteLine("Reader exception: " + ex.ToString()); Console.ReadLine(); }
            int[] result = Array.ConvertAll(b, Convert.ToInt32);
            LabelOffset++;
            fs.Close();
            foreach (int i in result) { return i; }
            return -1;
        }
        //Read a matrix from a file offset by two bytes of metadata
        static double[] ReadNextImage()
        {
            //Singleton
            if (ImageReaderRunning) { throw new Exception("Already accessing file"); }

            //Read image
            FileStream fs = File.OpenRead(ImagePath);
            //Reset parameters and decrement NN hyperparameters upon new epoch (currently disabled)
            if (!(ImageOffset < fs.Length)) { ImageOffset = 16; LabelOffset = 8; }
            fs.Position = ImageOffset;
            byte[] b = new byte[Resolution * Resolution];
            try
            {
                fs.Read(b, 0, Resolution * Resolution);
            }
            catch (Exception ex) { Console.WriteLine("Reader exception: " + ex.ToString()); Console.ReadLine(); }
            fs.Close();
            int[] array = Array.ConvertAll(b, Convert.ToInt32);
            ImageOffset += Resolution * Resolution;
            //Convert to 2d array
            double[] result = new double[Resolution * Resolution];
            //Convert array to doubles and store in result
            for (int i = 0; i < Resolution * Resolution; i++)
            {
                result[i] = array[i];
            }
            return result;
        }
        /// <summary>
        /// Returns a NN from a file
        /// </summary>
        /// <param name="GorD">Whether the NN is the generator (true) or discriminator (false)</param>
        /// <returns></returns>
        public static NN Read(bool GorD)
        {
            NN nn = new NN();
            nn.Layers = new List<iLayer>();
            string[] text;
            using (StreamReader sr = File.OpenText(GorD ? DWBPath : GWBPath))
            {
                text = sr.ReadToEnd().Split(',');
            }
            nn.NumLayers = int.Parse(text[0]);
            int iterator = 1;
            for (int i = 0; i < nn.NumLayers; i++)
            {
                bool fclORcl = int.Parse(text[iterator]) == 0; iterator++;
                int LayerCount = int.Parse(text[iterator]); iterator++;
                int InputLayerCount = int.Parse(text[iterator]); iterator++;

                if (fclORcl) { nn.Layers.Add(new FullyConnectedLayer(LayerCount, InputLayerCount)); }
                else { nn.Layers.Add(new ConvolutionLayer(LayerCount, InputLayerCount)); }

                for (int j = 0; j < nn.Layers[i].Weights.GetLength(0); j++)
                {
                    for (int jj = 0; jj < nn.Layers[i].Weights.GetLength(1); jj++)
                    {
                        nn.Layers[i].Weights[j, jj] = double.Parse(text[iterator]); iterator++;
                    }
                    if (i != nn.NumLayers - 1 && nn.Layers[i] is FullyConnectedLayer) 
                    { (nn.Layers[i] as FullyConnectedLayer).Biases[j] = double.Parse(text[iterator]); iterator++; }
                }
            }
            return nn;
        }
        /// <summary>
        /// Saves a specified NN to a file
        /// </summary>
        /// <param name="nn">The specified NN</param>
        /// <param name="GorD">Whether the NN is the generator (true) or discriminator (false)</param>
        public static void Write(NN nn, bool GorD)
        {
            StreamWriter sw = new StreamWriter(new FileStream(GorD ? GWBPath : DWBPath, FileMode.Create, FileAccess.Write, FileShare.None));
            sw.Write(nn.NumLayers + ",");
            for (int i = 0; i < nn.NumLayers; i++)
            {
                sw.Write((nn.Layers[i] is FullyConnectedLayer ? 0 : 1) + "," 
                    + nn.Layers[i].Length + "," + nn.Layers[i].InputLength + ",");
                for (int j = 0; j < nn.Layers[i].Weights.GetLength(0); j++)
                {
                    for (int jj = 0; jj < nn.Layers[i].Weights.GetLength(1); jj++)
                    {
                        sw.Write(nn.Layers[i].Weights[j, jj] + ",");
                    }
                    if (i != nn.NumLayers - 1 && nn.Layers[i] is FullyConnectedLayer)
                    { sw.Write((nn.Layers[i] as FullyConnectedLayer).Biases[j] + ","); }
                }
            }
            sw.Close();
        }
    }
}
