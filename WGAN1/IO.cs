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
            using (StreamReader sr = new StreamReader(new FileStream(LabelPath, FileMode.Open) { Position = LabelOffset }))
            {
                while (sr.Read() != number)
                {
                    if (sr.EndOfStream)
                    { 
                        LabelOffset = 8;
                        sr.BaseStream.Position = LabelOffset;
                        ImageOffset = 16;
                        Reset = true;
                    }
                    LabelOffset++;
                    ImageOffset += 784;
                }
                LabelOffset++;
            }
            double[] output;
            using (FileStream fs = new FileStream(ImagePath, FileMode.Open) { Position = ImageOffset })
            {
                if (fs.Length < fs.Position + 784)
                {
                    LabelOffset = 8;
                    fs.Position = ImageOffset;
                    ImageOffset = 16;
                    Reset = true;
                }
                byte[] bytes = new byte[Resolution * Resolution];
                fs.Read(bytes, 0, Resolution * Resolution);
                var ints = Array.ConvertAll(bytes, Convert.ToInt32);
                output = Array.ConvertAll(ints, Convert.ToDouble);
                ImageOffset += 784;
            }
            return output;
        }

        /* 
         * Apologies that this code is indecipherable; it was made during a flury of changes and bugfixes.
         * Documenting it would have been purposeless at the time of writing due to how quickly it was being changed.
         * And it's too large to easily document now.
         * 
         * Its purpose is to read/write NNs to a file in CSV format
         * If you misalign something, it will silently fail
         * 
         * Should you choose to ignore this warning, good luck to you, a brave soul!
         */

        /// <summary>
        /// Returns a NN from a file
        /// </summary>
        /// <param name="COG">[C]ritic [O]r [G]enerator</param>
        /// <returns></returns>
        public static NN Read(bool COG)
        {
            NN nn = new NN();
            nn.Layers = new List<Layer>();
            nn.ResidualLayers = new List<bool>();
            nn.BatchNormLayers = new List<bool>();
            nn.Activations = new List<int>();
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
                bool downorup = false;
                if (type == "2")
                { 
                    kernelsize = int.Parse(text[iterator]); iterator++;
                    padsize = int.Parse(text[iterator]); iterator++;
                    stride = int.Parse(text[iterator]); iterator++;
                    if (text[iterator] == "1") { downorup = true; } iterator++;
                }
                if (type == "3")
                {
                    downorup = text[iterator] == "1"; iterator++;
                    kernelsize = int.Parse(text[iterator]); iterator++;
                }
                int LayerCount = int.Parse(text[iterator]); iterator++;
                int InputLayerCount = int.Parse(text[iterator]); iterator++;
                nn.ResidualLayers.Add(text[iterator] == "1"); iterator++;
                nn.BatchNormLayers.Add(text[iterator] == "1"); iterator++;
                nn.Activations.Add(int.Parse(text[iterator])); iterator++;

                if (type == "0") { nn.Layers.Add(new FullyConnectedLayer(LayerCount, InputLayerCount)); nn.Layers[i].ActivationFunction = nn.Activations[i]; }
                //No weights exist in a sum layer
                if (type == "1") { nn.Layers.Add(new SumLayer(LayerCount, InputLayerCount)); nn.Layers[i].ActivationFunction = nn.Activations[i]; continue; }
                if (type == "2")
                {
                    nn.Layers.Add(new ConvolutionLayer(kernelsize, InputLayerCount));
                    var conv = nn.Layers[i] as ConvolutionLayer;
                    nn.Layers[i].Length = LayerCount;
                    nn.Layers[i].ActivationFunction = nn.Activations[i];
                    conv.PadSize = padsize; conv.Stride = stride; conv.DownOrUp = downorup;
                }
                if (type == "3")
                {
                    nn.Layers.Add(new PoolingLayer(downorup, kernelsize, InputLayerCount));
                    nn.Layers[i].ActivationFunction = nn.Activations[i];
                    //No weights exist in a pooling layer
                    continue;
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
                    var conv = nn.Layers[i] as ConvolutionLayer;
                    sw.Write("2," + conv.KernelSize.ToString() + ","+ conv.PadSize.ToString() 
                        + "," + conv.Stride.ToString() + "," + (conv.DownOrUp ? "1," : "0,"));
                }
                if (nn.Layers[i] is PoolingLayer)
                {
                    var pool = nn.Layers[i] as PoolingLayer;
                    sw.Write("3," + (pool.DownOrUp ? "1," : "0,") + pool.PoolSize + ",");
                }
                sw.Write(nn.Layers[i].Length + "," + nn.Layers[i].InputLength + ","
                    + (nn.ResidualLayers[i] ? "1," : "0,") + (nn.BatchNormLayers[i] ? "1," : "0,")
                    + nn.Activations[i].ToString() + ",");
                //Sum layers have no weights
                if (nn.Layers[i] is SumLayer || nn.Layers[i] is PoolingLayer) { continue; }
                for (int j = 0; j < nn.Layers[i].Weights.GetLength(0); j++)
                {
                    for (int jj = 0; jj < nn.Layers[i].Weights.GetLength(1); jj++)
                    {
                        sw.Write(Math.Round(nn.Layers[i].Weights[j, jj], 5) + ",");
                    }
                    if (i != nn.NumLayers - 1 && nn.Layers[i] is FullyConnectedLayer)
                    { sw.Write(Math.Round((nn.Layers[i] as FullyConnectedLayer).Biases[j], 5) + ","); }
                }
            }
            sw.Close();
        }
    }
}
