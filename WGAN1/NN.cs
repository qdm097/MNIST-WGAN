using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;

namespace WGAN1
{
    class NN
    {
        public int NumLayers { get; set; }
        public List<iLayer> Layers { get; set; }
        public static double LearningRate = 0.00005;
        public static double RMSDecay = .9;
        public static bool UseRMSProp = true;
        public static bool UseClipping = false;
        public static double ClipParameter = 10;
        public double BatchSize { get; set; }
        public static bool Training = false;
        public static bool Clear = false;
        public static bool Save = true;
        int Trials = 0;
        public double Error = 0;

        /// <summary>
        /// Generates a new NN with the specified parameters (using LeCun initialization)
        /// </summary>
        /// <param name="l">Number of layers in the network</param>
        /// <param name="wcs">Number of weights/biases in the network</param>
        public NN Init(List<iLayer> layers)
        {
            Layers = layers;
            NumLayers = Layers.Count;
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Init(i == NumLayers -1);
            }
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
        /// <param name="m">Batch size</param>
        /// <param name="ctg">Critic to generator ratio</param>
        /// <param name="num">What number is being generated</param>
        /// <param name="LatentSize">The size of the latent space for the generator</param>
        /// <param name="activeform">The form where the image will be updated</param>
        /// <param name="imgspeed">How quickly the image should update as a function of the algorithm</param>
        public static void Train(NN Critic, NN Generator, int LatentSize, int resolution,
             int m, int ctg, int num, Form1 activeform, int imgspeed, bool inputnorm, bool gradientnorm)
        {
            int formupdateiterator = 0;
            //The generator of the latentspace
            Random r = new Random();
            
            while (Training)
            {
                double totalrealmean = 0;
                double totalrealstddev = 0;
                //Train critic x times per 1 of generator
                //The generator's Wassersetin Loss
                double AvgFakeScore = 0;
               
                for (int i = 0; i < ctg; i++)
                {
                    //The critic's Wasserstein Loss
                    double CWLoss = 0;
                    //Batch norm stuff
                    double realmean = 0;
                    double realstddev = 0;

                    //Generate samples
                    var realsamples = new List<double[]>();
                    var fakesamples = new List<double[]>();
                    for (int ii = 0; ii < m; ii++)
                    {
                        //Generate fake image from latent space
                        fakesamples.Add(Generator.GenerateSample(Maths.RandomGaussian(r, LatentSize), inputnorm));
                        //Find next image
                        realsamples.Add(IO.FindNextNumber(num));
                        //Calculate values to help scale the fakes
                        var mean = Maths.CalcMean(realsamples[ii]);
                        realmean += mean;
                        realstddev += Maths.CalcStdDev(realsamples[ii], mean);
                    }
                    //Batch normalization
                    realmean /= m; totalrealmean += realmean;
                    realstddev /= m; totalrealstddev += realstddev;
                    for (int ii = 0; ii < m; ii++)
                    {
                        realsamples[ii] = Maths.Normalize(realsamples[ii]);
                        //realsamples[ii] = Maths.Normalize(realsamples[ii], realmean, realstddev);
                    }

                    double overallscore = 0;
                    //Critic's scores of each type of sample
                    List<double> rscores = new List<double>();
                    List<double> fscores = new List<double>();
                    //Compute values and loss
                    //Wasserstein loss = Avg(CScore(real) - CScore(fake))
                    for (int j = 0; j < m; j++)
                    {
                        //Real image
                        Critic.Calculate(realsamples[j], inputnorm);
                        rscores.Add(Critic.Layers[Critic.NumLayers - 1].Values[0]);
                        CWLoss += rscores[j];
                        //Fake image
                        Critic.Calculate(fakesamples[j], inputnorm);
                        fscores.Add(Critic.Layers[Critic.NumLayers - 1].Values[0]);
                        CWLoss -= fscores[j];
                        AvgFakeScore += fscores[j];
                    }
                    CWLoss /= 2d * m;
                    double GError = 0;
                    //Backprop
                    for (int j = 0; j < m; j++)
                    {
                        Critic.CalcGradients(realsamples[j], CWLoss, null, true);
                        overallscore += Critic.Layers[Critic.NumLayers - 1].Values[0] > 0 ? 1 : 0;
                        Critic.CalcGradients(fakesamples[j], -CWLoss, null, true);
                        overallscore += Critic.Layers[Critic.NumLayers - 1].Values[0] < 0 ? 1 : 0;
                        GError += Critic.Layers[Critic.NumLayers - 1].Values[0] > 0 ? 1 : 0;
                    }
                    //Update WBs
                    Critic.Update(m, gradientnorm);

                    //Report values to the front end
                    if (Clear) { Critic.Trials = 0; Generator.Trials = 0; Clear = false; }
                    overallscore /= (2 * m);
                    GError /= m;
                    Critic.Trials++;
                    Generator.Trials++;
                    Critic.Error = (Critic.Error * ((Critic.Trials) / (Critic.Trials + 1d))) + (overallscore * (1d / (Critic.Trials)));
                    Generator.Error = (Generator.Error * ((Generator.Trials) / (Generator.Trials + 1d))) + (GError * (1d / (Generator.Trials)));
                }
                //Adjust loss for batch size and critic to generator ratio
                AvgFakeScore /= m * ctg;

                //Train generator
                double[] test = new double[resolution * resolution];
                for (int i = 0; i < m; i++)
                {
                    var latentspace = Maths.RandomGaussian(r, LatentSize);
                    test = Generator.GenerateSample(latentspace, inputnorm);
                    Critic.Calculate(test, gradientnorm);
                    //Backprop through the critic to the generator
                    Critic.CalcGradients(test, -AvgFakeScore, null, false);
                    Generator.CalcGradients(latentspace, -AvgFakeScore, Critic.Layers[0], true);
                }
                Generator.Update(m, gradientnorm);
                //Update image (if applicable)
                if (formupdateiterator >= imgspeed)
                {
                    //Code that converts normalized generator outputs into an image
                    //Changes distribution of output values to 0-255 (brightness)
                    totalrealmean /= ctg; totalrealstddev /= ctg;
                    var values = Maths.Rescale(test, totalrealmean, totalrealstddev);
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
                        activeform.CScore = Critic.Error.ToString();
                        activeform.GScore = Generator.Error.ToString();
                        if (Critic.Error > Form1.Cutoff) { Training = false; }
                        if (IO.Reset)
                        {
                            IO.Reset = false;
                            activeform.Epoch++;
                        }
                    }); 
                    formupdateiterator = 0;
                }
                formupdateiterator++;
            }
            if (Save)
            {
                //Save nns
                IO.Write(Generator, false);
                IO.Write(Critic, true);
            }
            activeform.Invoke((Action)delegate
            {
                //Notify of being done training
                activeform.DoneTraining = true;
                //Reset errors
                activeform.CScore = null;
                activeform.GScore = null;
            });
        }
        public void Calculate(double[] input, bool batchnorm)
        {
            //Calculate
            Layers[0].Calculate(input, false, batchnorm);
            for (int i = 1; i < NumLayers; i++)
            {
                if (batchnorm)
                {
                    Layers[i].Calculate(Maths.Normalize(Layers[i - 1].Values), i == NumLayers - 1, false);
                }
                else
                {
                    Layers[i].Calculate(Layers[i - 1].Values, i == NumLayers - 1, batchnorm);
                }
            }
        }
        /// <summary>
        /// Backpropegate the error, determine the gradients
        /// </summary>
        /// <param name="input">The input of the network</param>
        /// <param name="loss">The loss of the NN</param>
        public void CalcGradients(double[] input, double correct, iLayer critic, bool calcgradients)
        {
            //Backprop
            for (int i = NumLayers - 1; i >= 0; i--)
            {
                bool isoutput; iLayer outputlayer;
                //If the output layer is also the critic's
                //It backprops off of the correctness of its identification of the sample
                if (critic is null) 
                { 
                    isoutput = i == Layers.Count - 1; 
                    outputlayer = isoutput ? null : Layers[i + 1];
                }
                //Otherwise it backprops off of the critic's interpretaton of the sample
                else
                {
                    isoutput = false;
                    outputlayer = i == Layers.Count - 1 ? critic : Layers[i + 1];  
                }
                double[] inputvals = i == 0 ? input : Layers[i - 1].Values;
                Layers[i].Backprop(inputvals, outputlayer, isoutput, correct, calcgradients);
            }
        }
        /// <summary>
        /// Updates the NN's layer's weights after a full batch of gradient descent
        /// </summary>
        /// <param name="m">Batch size</param>
        /// <param name="a">Learning rate</param>
        /// <param name="c">Clipping parameter</param>
        /// <param name="rmsd">RMSProp decay parameter</param>
        public void Update(int m, bool batchnorm)
        {
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Descend(m, batchnorm);
            }
        }
        double[] GenerateSample(double[] latentspace, bool batchnorm)
        {
            double[] image = latentspace;
            for (int i = 0; i < NumLayers; i++)
            {
                if (batchnorm)
                {
                    Layers[i].Calculate(image, i == NumLayers - 1, false);
                    image = Maths.Normalize(Layers[i].Values);
                }
                else
                {
                    Layers[i].Calculate(image, i == NumLayers - 1, true);
                    image = Layers[i].Values;
                }
            }
            return image;
        }
    }
}
