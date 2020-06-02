using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Reflection.Emit;

namespace WGAN1
{
    static class IO
    {
        public static bool Testing = false;
        public static bool LabelReaderRunning = false;
        public static bool ImageReaderRunning = false;
        static readonly string GWBPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\GeneratorWBs.txt";
        static readonly string CWBPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\DiscriminatorWBs.txt";
        static readonly string TrainImagePath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\train-images.idx3-ubyte";
        static readonly string TrainLabelPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\train-labels.idx1-ubyte";
        static readonly string TestLabelPath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\test-labels.idx1-ubyte";
        static readonly string TestImagePath = Directory.GetParent(Environment.CurrentDirectory).Parent.FullName + "\\test-images.idx3-ubyte";
        private static string LabelPath = Testing ? TestLabelPath : TrainLabelPath;
        private static string ImagePath = Testing ? TestImagePath : TrainImagePath;
        static int LabelOffset = 8;
        static int ImageOffset = 16;
        public static bool Reset = false;
        static int Resolution = 28;

        public static double[] FindNextNumber(int number)
        {
            //Find the next number
            while(ReadNextLabel() != number)
            {
                ImageOffset += 784; 
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
            if (!(LabelOffset < fs.Length)) { LabelOffset = 8; ImageOffset = 16; Reset = true; }

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
            if (!(ImageOffset < fs.Length)) { ImageOffset = 16; LabelOffset = 8; Reset = true; }
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
        /// <param name="COG">[C]ritic [O]r [G]enerator</param>
        /// <returns></returns>
        public static NN Read(bool COG)
        {
            NN nn = new NN();
            nn.Layers = new List<iLayer>();
            nn.ResidualLayers = new List<bool>();
            nn.BatchNormLayers = new List<bool>();
            nn.TanhLayers = new List<bool>();
            string[] text;
            using (StreamReader sr = File.OpenText(COG ? CWBPath : GWBPath))
            {
                text = sr.ReadToEnd().Split(',');
            }
            nn.NumLayers = int.Parse(text[0]);
            int iterator = 1;
            for (int i = 0; i < nn.NumLayers; i++)
            {
                string type = text[iterator]; iterator++;
                int kernelsize = 0;
                int padsize = 0;
                int stride = 0;
                if (type == "2")
                { 
                    kernelsize = int.Parse(text[iterator]); iterator++;
                    padsize = int.Parse(text[iterator]); iterator++;
                    stride = int.Parse(text[iterator]); iterator++;
                }
                int LayerCount = int.Parse(text[iterator]); iterator++;
                int InputLayerCount = int.Parse(text[iterator]); iterator++;
                nn.ResidualLayers.Add(text[iterator] == "1"); iterator++;
                nn.BatchNormLayers.Add(text[iterator] == "1"); iterator++;
                nn.TanhLayers.Add(text[iterator] == "1"); iterator++;

                if (type == "0") { nn.Layers.Add(new FullyConnectedLayer(LayerCount, InputLayerCount)); }
                //No weights exist in a sum layer
                if (type == "1") { nn.Layers.Add(new SumLayer(LayerCount, InputLayerCount)); continue; }
                if (type == "2")
                { 
                    nn.Layers.Add(new ConvolutionLayer(kernelsize, InputLayerCount)); 
                    nn.Layers[i].Length = LayerCount;
                    (nn.Layers[i] as ConvolutionLayer).PadSize = padsize;
                    (nn.Layers[i] as ConvolutionLayer).Stride = stride;
                }

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
        /// <param name="COG">[C]ritic [O]r [G]enerator</param>
        public static void Write(NN nn, bool COG)
        {
            StreamWriter sw = new StreamWriter(new FileStream(COG ? CWBPath : GWBPath, FileMode.Create, FileAccess.Write, FileShare.None));
            sw.Write(nn.NumLayers + ",");
            for (int i = 0; i < nn.NumLayers; i++)
            {
                if (nn.Layers[i] is FullyConnectedLayer)
                {
                    sw.Write("0,");
                }
                if (nn.Layers[i] is SumLayer)
                {
                    sw.Write("1,");
                }
                if (nn.Layers[i] is ConvolutionLayer)
                {
                    sw.Write("2," + (nn.Layers[i] as ConvolutionLayer).KernelSize.ToString() + ","
                        + ((nn.Layers[i] as ConvolutionLayer).PadSize.ToString()) + ","
                        + (nn.Layers[i] as ConvolutionLayer).Stride.ToString() + ",");
                }
                sw.Write(nn.Layers[i].Length + "," + nn.Layers[i].InputLength + ","
                    + (nn.ResidualLayers[i] ? "1" : "0") + "," + (nn.BatchNormLayers[i] ? "1" : "0") + ","
                    + (nn.TanhLayers[i] ? "1" : "0") + ",");
                //Sum layers have no weights
                if (nn.Layers[i] is SumLayer) { continue; }
                for (int j = 0; j < nn.Layers[i].Weights.GetLength(0); j++)
                {
                    for (int jj = 0; jj < nn.Layers[i].Weights.GetLength(1); jj++)
                    {
                        sw.Write(Math.Round(nn.Layers[i].Weights[j, jj], 4) + ",");
                    }
                    if (i != nn.NumLayers - 1 && nn.Layers[i] is FullyConnectedLayer)
                    { sw.Write(Math.Round((nn.Layers[i] as FullyConnectedLayer).Biases[j], 4) + ","); }
                }
            }
            sw.Close();
        }
    }
}
