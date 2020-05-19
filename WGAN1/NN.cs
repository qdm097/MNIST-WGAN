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
        int ConvLayerPoint { get; set; }
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
        public static NN GenerateNN(int l, List<int> wcs, int inputsize, bool iscnn)
        {
            if (l != wcs.Count()) { throw new Exception("Invalid wc to l ratio"); }
            NN nn = new NN();
            nn.NumLayers = l;
            nn.LayerCounts = wcs;
            nn.Layers = new List<Layer>();
            nn.ConvLayerPoint = 0;
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
        /// <param name="convlayerpoint">At what point the generator's neurons are after the convolutional layer</param>
        public static void Train(bool LoadOrGenerate, int clcount, int glcount, List<int> cwbcount, List<int> gwbcount,
            int resolution, double a, double c, int m, int ctg, double rmsd, int num, Form1 activeform, int imgspeed, int convlayerpoint)
        {
            NN Critic;
            NN Generator;
            if (LoadOrGenerate) { Critic = IO.Read(true); Generator = IO.Read(false); }
            else 
            {
                Critic = GenerateNN(clcount, cwbcount, resolution * resolution, false).SetHyperParams(a, c);
                //Generator does not have clipping
                Generator = GenerateNN(glcount, gwbcount, LatentSize, true).SetHyperParams(a, 99);
                Generator.ConvLayerPoint = convlayerpoint;
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
                        fakesamples.Add(Generator.GenerateSample(Statistics.RandomGaussian(LatentSize)));
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
                        overallscore += Critic.Layers[Critic.NumLayers - 1].Values[0] > 0 ? 1 : 0;
                        //Fake image
                        Critic.Calculate(fakesamples[j]);
                        Critic.CalcGradients(fakesamples[i], fakeanswer, null);
                        overallscore += Critic.Layers[Critic.NumLayers - 1].Values[0] < 0 ? 1 : 0;
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
                    var latentspace = Statistics.RandomGaussian(LatentSize);
                    test = Generator.GenerateSample(latentspace);
                    Critic.Layers[0].Calculate(test, false);
                    for (int jj = 1; jj < Critic.NumLayers; jj++)
                    {
                        Critic.Layers[jj].Calculate(Critic.Layers[jj - 1].Values, jj == Critic.NumLayers - 1);
                    }
                    //Backprop critic layer
                    for (int jj = Critic.NumLayers - 1; jj >= 0; jj--)
                    {
                        //If an output layer
                        if (jj == Critic.NumLayers - 1)
                        { 
                            Critic.Layers[Critic.NumLayers - 1].Backprop(-1); 
                        }
                        else 
                        { 
                            Critic.Layers[jj].Backprop(Critic.Layers[jj + 1]);
                        }
                    }
                    Generator.CalcGradients(latentspace, -1, Critic.Layers[0]);
                }
                Generator.Update(m, a, c, rmsd);
                //Update image (if applicable)
                if (imgupdateiterator >= imgspeed)
                {
                    //Code that converts normalized generator outputs into an image
                    //Changes distribution of output values to 0-255 (brightness)
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
        public void CalcGradients(double[] input, double loss, Layer critic)
        {
            //Backpropegate post conv layers
            for (int jj = NumLayers - 1; jj >= (ConvLayerPoint != 0 ? ConvLayerPoint - 1 : 0); jj--)
            {
                //If not an output layer
                if (jj != NumLayers - 1) { Layers[jj].Backprop(Layers[jj + 1]); continue; }
                if (loss != 0) { Layers[jj].Backprop(loss); continue; }
                //Backprop generator's errors from the critic
                if (!(critic is null)) { Layers[jj].Backprop(critic); continue; }
                throw new Exception("Invalid inputs");
            }

            //Backprop conv layer
            if (!(ConvLayer is null))
            {
                if (Layers.Count == 0 || ConvLayerPoint > NumLayers) { ConvLayer.Backprop(critic); }
                else { ConvLayer.Backprop(Layers[ConvLayerPoint - 1]); }
                ConvLayer.Descend(input);
            }

            //Backprop preconv layers
            if (ConvLayerPoint != 0) { Layers[(NumLayers > ConvLayerPoint ? ConvLayerPoint : NumLayers) - 1].Backprop(ConvLayer); }
            if (ConvLayerPoint > 1)
            {
                //Backprop preconv layers
                for (int jj = (NumLayers > ConvLayerPoint ? ConvLayerPoint : NumLayers) - 2; jj >= 0; jj--)
                {
                    Layers[jj].Backprop(Layers[jj + 1]);
                }
            }
            //Descend
            if (ConvLayerPoint != 0)
            {
                //Preconv layers
                for (int jj = 0; jj < (NumLayers > ConvLayerPoint ? ConvLayerPoint : NumLayers); jj++)
                {
                    if (jj == 0) { Layers[jj].Descend(input, false); }
                    else { Layers[jj].Descend(Layers[jj - 1].Values, false); }
                }
                //Postconv layers
                for (int jj = ConvLayerPoint; jj < NumLayers; jj++)
                {
                    if (jj == ConvLayerPoint) { Layers[jj].Descend(ConvLayer.ZVals); }
                    else { Layers[jj].Descend(Layers[jj - 1].Values, jj == NumLayers - 1); }
                }
            }
            else
            {
                for (int jj = 0; jj < NumLayers; jj++)
                {
                    if (jj == 0) { Layers[jj].Descend(input, false); }
                    else { Layers[jj].Descend(Layers[jj - 1].Values, jj == NumLayers - 1); }
                }
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
            double[] image = latentspace;
            //Return convolved image if there are no layers
            if (Layers.Count == 0) { return ConvLayer.CalcDotProduct(image); }
            //Layers to determine input space
            if (ConvLayerPoint != 0) {
                for (int i = 0; i < (NumLayers > ConvLayerPoint ? ConvLayerPoint : NumLayers); i++)
                {
                    Layers[i].Calculate(image, false);
                    image = Layers[i].Values;
                }
            }
            image = ConvLayer.CalcDotProduct(image);
            //Layers to modify output image
           
            for (int i = ConvLayerPoint; i < NumLayers; i++)
            {
                if (i == ConvLayerPoint) { Layers[ConvLayerPoint].Calculate(image, false); }
                else { Layers[i].Calculate(Layers[i - 1].Values, i == NumLayers - 1); }
            }
            return image;
        }
    }
}
