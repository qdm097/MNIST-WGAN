using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;

namespace WGAN1
{
    class NN
    {
        public int NumLayers { get; set; }
        public List<int> LayerCounts { get; set; }
        public static int LatentSize = 28;
        public static int ConvXSize = 28 * 28;
        public Convolution ConvLayer { get; set; }
        public List<Layer> Layers { get; set; }
        public double LearningRate { get; set; }
        public double ClippingParameter { get; set; }
        public double BatchSize { get; set; }
        public static bool Training = false;
        int trials = 0;
        public double PercCorrect = 0;

        /// <summary>
        /// Generates a new NN with the specified parameters (using LeCun initialization)
        /// </summary>
        /// <param name="l">Number of layers in the network</param>
        /// <param name="wcs">Number of weights/biases in the network</param>
        public NN GenerateNN(int l, List<int> wcs, int inputsize, bool iscnn)
        {
            if (l != wcs.Count()) { throw new Exception("Invalid wc to l ratio"); }
            NN nn = new NN();
            nn.NumLayers = l;
            nn.LayerCounts = wcs;
            nn.Layers = new List<Layer>();
            var r = new Random();
            if (iscnn)
            {
                nn.ConvLayer = new Convolution(ConvXSize, LatentSize).Init(ConvXSize, LatentSize);
            }
            for (int i = 0; i < l; i++)
            {
                //Input layer has input count of resolution of image (28*28)
                nn.Layers.Add(new Layer(wcs[i], i == 0 ? inputsize : wcs[i - 1]));
                //All layers have weights
                nn.Layers[i].Weights = new double[nn.Layers[i].Length, nn.Layers[i].InputLength];
                //Output layer has no biases
                if (i != l - 1) { nn.Layers[i].Biases = new double[nn.Layers[i].Length]; }
                //Initialize weights (and biases to zero)
                for (int j = 0; j < wcs[i]; j++)
                {
                    for (int jj = 0; jj < nn.Layers[i].InputLength; jj++)
                    {
                        nn.Layers[i].Weights[j, jj] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (nn.Layers[i].InputLength * nn.Layers[i].InputLength));
                    }
                }
            }
            return nn;
        }
        /// <summary>
        /// Sets the hyper parameters of the NN
        /// </summary>
        /// <param name="a">Learning rate</param>
        /// <param name="c">Clipping parameter</param>
        /// <returns></returns>
        public NN SetHyperParams(double a, double c)
        {
            LearningRate = a; ClippingParameter = c;
            return this;
        }
        /// <summary>
        /// Trains the GAN
        /// </summary>
        /// <param name="LoadOrGenerate">Whether to load the WBs or to generate new ones</param>
        /// <param name="clcount">How many layers are in the critic</param>
        /// <param name="glcount">How many layers are in the generator</param>
        /// <param name="cwbcount">How many WBs are in the critic per layer</param>
        /// <param name="gwbcount">How many WBs are in the generator per layer</param>
        /// <param name="a">Learning rate</param>
        /// <param name="c">Clipping parameter</param>
        /// <param name="m">Batch size</param>
        /// <param name="ctg">Critic to generator ratio</param>
        /// <param name="rmsd">How quickly the RMS gradients decay (psuedo momentum)</param>
        /// <param name="num">What number is being generated</param>
        /// <param name="LatentSize">The size of the latent space for the generator</param>
        /// <param name="activeform">The form where the image will be updated</param>
        /// <param name="imgspeed">How quickly the image should update as a function of the algorithm</param>
        public static void Train(bool LoadOrGenerate, int clcount, int glcount, List<int> cwbcount, List<int> gwbcount,
            int resolution, double a, double c, int m, int ctg, double rmsd, int num, Form1 activeform, int imgspeed)
        {
            NN Critic;
            NN Generator;
            if (LoadOrGenerate) { Critic = IO.Read(true); Generator = IO.Read(false); }
            else 
            {
                Critic = new NN().SetHyperParams(a, c).GenerateNN(clcount, cwbcount, resolution * resolution, false);
                //Generator does not have clipping
                Generator = new NN().SetHyperParams(a, 99).GenerateNN(glcount, gwbcount, LatentSize, true);
            }
            int imgupdateiterator = 0;
            while (Training)
            {
                //Train critic x times per 1 of generator
                for (int i = 0; i < ctg; i++)
                {
                    //Generate samples
                    var realsamples = new List<double[]>();
                    var fakesamples = new List<double[]>();
                    for (int ii = 0; ii < m; ii++)
                    {
                        //Generate latent space
                        fakesamples.Add(Generator.GenerateSample(Statistics.RandomGaussian(LatentSize)));
                        //Find next image
                        realsamples.Add(IO.FindNextNumber(num));
                    }
                    double realanswer = 1d;
                    double fakeanswer = -1;
                    double overallscore = 0;
                    for (int j = 0; j < m; j++)
                    {
                        //Need to implement Wasserstein Loss = real score - fake score

                        Critic.Calculate(realsamples[j]);
                        overallscore += Critic.Layers[Critic.NumLayers - 1].Values[0] > 0 ? 1 : 0;
                        Critic.Calculate(fakesamples[j]);
                        overallscore += Critic.Layers[Critic.NumLayers - 1].Values[0] < 0 ? 1 : 0;
                        Critic.CalcGradients(realsamples[i], realanswer, null);
                        Critic.CalcGradients(fakesamples[i], fakeanswer, null);
                    }
                    overallscore /= 2 * m;
                    Critic.PercCorrect = overallscore;
                    Critic.Update(m, a, c, rmsd);
                }
                //Train generator
                double[] test = new double[resolution * resolution];
                for (int i = 0; i < m; i++)
                {
                    var latentspace = Statistics.RandomGaussian(LatentSize);
                    test = Generator.GenerateSample(latentspace);
                    Critic.Layers[0].Calculate(test, false);
                    for (int jj = 1; jj < Critic.NumLayers; jj++)
                    {
                        Critic.Layers[jj].Calculate(Critic.Layers[jj - 1].Values, jj == Critic.NumLayers - 1);
                    }
                    Generator.CalcGradients(latentspace, 0, Critic.Layers[0]);
                }
                Generator.Update(m, a, c, rmsd);

                //Code that converts normalized generator outputs into an image
                //Change distribution of output values to 0-255 (brightness)
                var values = Statistics.Rescale(test, 0, 255);
                var image = new int[resolution, resolution];
                int iterator = 0;
                //Convert values to a 2d array
                for (int i = 0; i < resolution; i++)
                {
                    for (int ii = 0; ii < resolution; ii++)
                    {
                        image[ii, i] = (int)values[iterator]; iterator++;
                    }
                }
                if (imgupdateiterator >= imgspeed)
                { 
                    activeform.Invoke((Action)delegate { activeform.image = image; activeform.CScore = Critic.PercCorrect.ToString(); }); imgupdateiterator = 0;
                }
                imgupdateiterator++;
            }
            activeform.Invoke((Action)delegate { activeform.DoneTraining = true; });
            //Save nns
            IO.Write(Generator, true);
            IO.Write(Critic, false);
        }
        public void Calculate(double[] input)
        {
            //Calculate
            Layers[0].Calculate(input, false);
            for (int jj = 1; jj < NumLayers; jj++)
            {
                Layers[jj].Calculate(Layers[jj - 1].Values, jj == NumLayers - 1);
            }
        }
        /// <summary>
        /// Backpropegate the error, determine the gradients
        /// </summary>
        /// <param name="input">The input of the network</param>
        /// <param name="loss">The loss of the NN</param>
        public void CalcGradients(double[] input, double loss, Layer critic)
        {
            //Backpropegate
            for (int jj = NumLayers - 1; jj >= 0; jj--)
            {
                //If not an output layer
                if (jj != NumLayers - 1) { Layers[jj].Backprop(Layers[jj + 1]); continue; }
                if (loss != 0) { Layers[jj].Backprop(loss); continue; }
                //Backprop generator's errors from the critic
                if (!(critic is null)) { Layers[jj].Backprop(critic); continue; }
                throw new Exception("Invalid inputs");
            }
            if (!(ConvLayer is null))
            {
                if (Layers.Count == 0) { ConvLayer.Backprop(critic); }
                else { ConvLayer.Backprop(Layers[0]); }
                ConvLayer.Descend(input);
            }
            //Descend
            for (int jj = 0; jj < NumLayers; jj++)
            {
                if (jj == 0) { Layers[jj].Descend(input, false); }
                else { Layers[jj].Descend(Layers[jj - 1].Values, jj == NumLayers - 1); }
            }
        }
        /// <summary>
        /// Updates the NN's layer's weights after a full batch of gradient descent
        /// </summary>
        /// <param name="m">Batch size</param>
        /// <param name="a">Learning rate</param>
        /// <param name="c">Clipping parameter</param>
        /// <param name="rmsd">RMSProp decay parameter</param>
        public void Update(int m, double a, double c, double rmsd)
        {
            if (!(ConvLayer is null)) { ConvLayer.Descend(m, a, c, rmsd); }
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Descend(m, a, c, rmsd);
            }
        }
        double[] GenerateSample(double[] latentspace)
        {
            var image = ConvLayer.CalcDotProduct(latentspace);
            if (Layers.Count > 0) { Layers[0].Calculate(image, false); }
            for (int i = 1; i < NumLayers; i++)
            {
                Layers[i].Calculate(Layers[i - 1].Values, i == NumLayers - 1);
            }
            //Normalize resulting matrix (or convolution's matrix if there are no other layers)
            return Statistics.Normalize(Layers.Count > 0 ? Layers[Layers.Count - 1].Values : image);
        }
    }
}
