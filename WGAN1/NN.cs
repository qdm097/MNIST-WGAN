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
        public List<iLayer> Layers { get; set; }
        public double LearningRate { get; set; }
        public double ClippingParameter { get; set; }
        public double BatchSize { get; set; }
        public static bool Training = false;
        public static bool Clear = false;
        int Trials = 0;
        public double PercCorrect = 0;

        /// <summary>
        /// Generates a new NN with the specified parameters (using LeCun initialization)
        /// </summary>
        /// <param name="l">Number of layers in the network</param>
        /// <param name="wcs">Number of weights/biases in the network</param>
        public static NN GenerateNN(int l, List<int> wcs, List<iLayer> layertypes, int inputsize)
        {
            if (l != wcs.Count()) { throw new Exception("Invalid wc to l ratio"); }
            NN nn = new NN();
            nn.NumLayers = l;
            nn.Layers = new List<iLayer>();
            var r = new Random();
            for (int i = 0; i < wcs.Count; i++)
            {
                if (layertypes[i] is FullyConnectedLayer)
                {
                    nn.Layers.Add(new FullyConnectedLayer(wcs[i], i == 0 ? inputsize : wcs[i - 1])
                        .Init(i == l - 1) as FullyConnectedLayer);
                }
                else
                {
                    nn.Layers.Add(new ConvolutionLayer(wcs[i], wcs[i])
                        .Init(false) as ConvolutionLayer);
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
        /// <param name="glayertypes">What type of layer each layer is (convolutional or fully connected)</param>
        /// <param name="clayertypes">Only feed in FCLs to this or things will break</param>
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
            List<iLayer> glayertypes, List<iLayer> clayertypes, int resolution, double a, double c, int m, int ctg, 
            double rmsd, int num, int LatentSize, Form1 activeform, int imgspeed)
        {
            NN Critic;
            NN Generator;
            if (LoadOrGenerate) { Critic = IO.Read(true); Generator = IO.Read(false); }
            else 
            {
                Critic = GenerateNN(clcount, cwbcount, clayertypes, resolution * resolution).SetHyperParams(a, c);
                //Generator does not have clipping
                Generator = GenerateNN(glcount, gwbcount, glayertypes, LatentSize).SetHyperParams(a, 99);
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
                        //Generate fake image from latent space
                        fakesamples.Add(Generator.GenerateSample(Maths.RandomGaussian(LatentSize)));
                        //Find next image
                        realsamples.Add(IO.FindNextNumber(num));
                    }
                    double realanswer = 1d;
                    double fakeanswer = -1d;
                    double overallscore = 0;
                    for (int j = 0; j < m; j++)
                    {
                        //Need to implement Wasserstein Loss = real score - fake score

                        //Real image
                        Critic.Calculate(realsamples[j]);
                        Critic.CalcGradients(realsamples[i], realanswer, null);
                        overallscore += (Critic.Layers[Critic.NumLayers - 1] as FullyConnectedLayer).Values[0] > 0 ? 1 : 0;
                        //Fake image
                        Critic.Calculate(fakesamples[j]);
                        Critic.CalcGradients(fakesamples[i], fakeanswer, null);
                        overallscore += (Critic.Layers[Critic.NumLayers - 1] as FullyConnectedLayer).Values[0] < 0 ? 1 : 0;
                    }
                    if (Clear) { Critic.Trials = 0; Critic.PercCorrect = 0; Clear = false; }
                    overallscore /= 2 * m;
                    double ratio = (double)Critic.Trials / (Critic.Trials + 1);
                    Critic.PercCorrect = (ratio * Critic.PercCorrect) + ((1 - ratio) * overallscore);
                    Critic.Trials++;
                    //Update WBs
                    Critic.Update(m, a, c, rmsd);
                }
                //Train generator
                double[] test = new double[resolution * resolution];
                for (int i = 0; i < m; i++)
                {
                    var latentspace = Maths.RandomGaussian(LatentSize);
                    test = Generator.GenerateSample(latentspace);
                    Critic.Calculate(test);
                    //Backprop critic layer
                    for (int jj = Critic.NumLayers - 1; jj >= 0; jj--)
                    {
                        //If an output layer
                        if (jj == Critic.NumLayers - 1)
                        { 
                            Critic.Layers[Critic.NumLayers - 1].Backprop(1); 
                        }
                        else 
                        { 
                            Critic.Layers[jj].Backprop(Critic.Layers[jj + 1]);
                        }
                    }
                    Generator.CalcGradients(latentspace, 0, Critic.Layers[0]);
                }
                Generator.Update(m, a, c, rmsd);
                //Update image (if applicable)
                if (imgupdateiterator >= imgspeed)
                {
                    //Code that converts normalized generator outputs into an image
                    //Changes distribution of output values to 0-255 (brightness)
                    var values = Maths.Rescale(test, 0, 255);
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
                    activeform.Invoke((Action)delegate
                    {
                        activeform.image = image; 
                        activeform.CScore = Critic.PercCorrect.ToString(); 
                    }); 
                    imgupdateiterator = 0;
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
        public void CalcGradients(double[] input, double loss, iLayer critic)
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
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Descend(m, a, c, rmsd);
            }
        }
        double[] GenerateSample(double[] latentspace)
        {
            double[] image = latentspace;
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Calculate(image, i == NumLayers - 1);
                image = Layers[i].Values;
            }
            return image;
        }
    }
}
