using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;
using System.Windows.Markup;
using System.Globalization;

namespace WGAN1
{
    class NN
    {
        public int NumLayers { get; set; }
        public List<Layer> Layers { get; set; }
        public List<int> Activations { get; set; }
        public List<bool> ResidualLayers { get; set; }
        public List<bool> BatchNormLayers { get; set; }
        List<double[]> Residuals { get; set; }
        public static double LearningRate = 0.000146;
        public static double RMSDecay = .9;
        
        //public static double Infinitesimal = 1E-20;
        public static bool UseClipping = false;
        public static double ClipParameter = 10;
        //Batch size used
        //Note: all for loops involving batchsize use 'b' as the iterator for clarity
        public static int BatchSize = 10;
        public static bool Training = false;
        public static bool Clear = false;
        public static bool Save = true;
        public static bool NormOutputs = false;
        public static bool NormErrors = false;
        public static bool UseRMSProp = true;
        public static double Infinitesimal = 1E-30;
        public int OutputLength { get; set; }
        int Trials = 0;
        public double Error = 0;
        public double PercCorrect = 0;

        /// <summary>
        /// Generates a new NN with the specified parameters (using LeCun initialization)
        /// </summary>
        /// <param name="l">Number of layers in the network</param>
        /// <param name="wcs">Number of weights/biases in the network</param>
        public NN Init(List<Layer> layers, List<int> activations, List<bool> residuals, List<bool> batchnorms)
        {
            Layers = layers;
            NumLayers = Layers.Count;
            Activations = activations;
            ResidualLayers = residuals;
            BatchNormLayers = batchnorms;
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Init(i == NumLayers - 1);
                Layers[i].ActivationFunction = Activations[i];
            }
            OutputLength = Layers[NumLayers - 1].OutputLength;
            return this;
        }
        public static void TestTrain (NN Critic, bool gradientnorm, int imgspeed, Form1 activeform)
        {
            int formupdateiterator = 0;

            //Test code to generate a new layer with predefined qualities

            //List<Layer> layers = new List<Layer>() { new ConvolutionLayer(4, 784) { DownOrUp = true, Stride = 1 }.Init(false), new ConvolutionLayer(3, 625){ DownOrUp = true, Stride = 1 }.Init(false),
            //    new ConvolutionLayer(2, 529){ DownOrUp = true, Stride = 1 }.Init(false), new FullyConnectedLayer(100, 484).Init(false), new FullyConnectedLayer(10, 100).Init(true) };
            //List<bool> tans = new List<bool>() { true, true, true, true, true};
            //List<bool> bns = new List<bool>() { false, false, false, false, false };
            //List<bool> ress = new List<bool>() { false, false, false, false, false };
            //NN Critic = new NN().Init(layers, tans, ress, bns);

            while (Training)
            {
                double mean = 0;
                double stddev = 0;
                double score = 0;
                double perccorrect = 0;
                List<List<double[]>> nums = new List<List<double[]>>();
                List<int> labels = new List<int>();
                Random r = new Random();
                for (int i = 0; i < 10; i++)
                {
                    var temp = new List<double[]>();
                    for (int j = 0; j < BatchSize; j++)
                    {
                        temp.Add(Maths.Normalize(IO.FindNextNumber(i)));
                        //var tmpmean = Maths.CalcMean(temp[j]);
                        //mean += tmpmean;
                        //stddev += Maths.CalcStdDev(temp[j], tmpmean);
                    }
                    nums.Add(temp);
                }

                //Batch normalization
                //mean /= 10 * batchsize; stddev /= 10 * batchsize;
                //for (int i = 0; i < 10; i++)
                //{
                //    nums[i] = Maths.BatchNormalize(nums[i], mean, stddev);
                //}
               
                //Foreach number
                for (int i = 0; i < 10; i++)
                {
                    Critic.Calculate(nums[i]);
                    //Foreach sample in the batch
                    for (int j = 0; j < BatchSize; j++)
                    {
                        double max = -99;
                        int guess = -1;
                        //Foreach output neuron
                        for (int k = 0; k < 10; k++)
                        {
                            var value = Critic.Layers[Critic.NumLayers - 1].Values[j][k];
                            score += Math.Pow(value - (k == i ? 1d : 0d), 2);
                            if (value > max) { max = value; guess = k; }
                        }
                        perccorrect += guess == i ? 1d : 0d;
                        labels.Add(guess);
                    }
                    Critic.CalcGradients(nums[i], null, i, true);
                }
              
                score /= (10 * BatchSize);
                perccorrect /= (10 * BatchSize);
                score = Math.Sqrt(score);

                Critic.Update(gradientnorm);

                //Report values to the front end
                if (Clear)
                {
                    Critic.Trials = 0; Critic.Error = 0; Critic.PercCorrect = 0; Clear = false;
                }
                
                Critic.Trials++;
                Critic.Error = (Critic.Error * ((Critic.Trials) / (Critic.Trials + 1d))) + (score * (1d / (Critic.Trials)));
                Critic.PercCorrect = (Critic.PercCorrect * ((Critic.Trials) / (Critic.Trials + 1d))) + (perccorrect * (1d / (Critic.Trials)));
               
                //Update image (if applicable)
                if (formupdateiterator >= imgspeed)
                {
                    //Maths.Rescale(list8[0], mean8, stddev8);
                    int index = r.Next(0, 10);
                    var values = Form1.Rescale(Maths.Convert(nums[index][0]));
                    var image = new int[28, 28];
                    //Convert values to a 2d array
                    for (int i = 0; i < 28; i++)
                    {
                        for (int ii = 0; ii < 28; ii++)
                        {
                            image[ii, i] = (int)values[i, ii];
                        }
                    }
                    activeform.Invoke((Action)delegate
                    {
                        activeform.image = image;
                        activeform.CScore = Critic.Error.ToString();
                        activeform.CPerc = Critic.PercCorrect.ToString();
                        //Critic.Layers[Critic.NumLayers - 1].Values[0][index].ToString();
                        activeform.Label = labels[index].ToString();
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
            activeform.Invoke((Action)delegate
            {
                //Notify of being done training
                activeform.DoneTraining = true;
                //Reset errors
                activeform.CScore = null;
                activeform.GScore = null;
            });
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
             int ctg, int num, Form1 activeform, int imgspeed, bool inputnorm, bool gradientnorm)
        {
            int formupdateiterator = 0;
            //The generator of the latentspace
            Random r = new Random();
            
            while (Training)
            {
                double totalrealmean = 0;
                double totalrealstddev = 0;
                //Train critic x times per 1 of generator
               
                for (int i = 0; i < ctg; i++)
                {
                    //Batch norm stuff
                    double realmean = 0;
                    double realstddev = 0;

                    double AvgRealScore = 0;
                    double AvgFakeScore = 0;

                    //Generate samples
                    var realsamples = new List<double[]>();
                    var latentspaces = new List<double[]>();
                    for (int ii = 0; ii < BatchSize; ii++)
                    {
                        
                        //Find next image
                        realsamples.Add(IO.FindNextNumber(num));

                        //Generate fake image from latent space
                        //fakesamples.Add(Generator.GenerateSample(Maths.RandomGaussian(r, LatentSize), inputnorm));
                        //Generate fake image from downscaled real image
                        latentspaces.Add(Maths.RandomGaussian(r, LatentSize));
                        //Calculate values to help scale the fakes
                        var mean = Maths.CalcMean(realsamples[ii]);
                        realmean += mean;
                        realstddev += Maths.CalcStdDev(realsamples[ii], mean);
                    }
                    realmean /= BatchSize; totalrealmean += realmean; 
                    realstddev /= BatchSize; totalrealstddev += realstddev;

                    //Batchnorm the samples
                    realsamples = Maths.BatchNormalize(realsamples, realmean, realstddev);
                    var fakesamples = Maths.BatchNormalize(Generator.GenerateSamples(latentspaces), realmean, realstddev);
                    //var fakesamples = Maths.BatchNormalize(Generator.GenerateSamples(latentspaces));

                    double overallscore = 0;
                    //Critic's scores of each type of sample
                    List<double> rscores = new List<double>();
                    List<double> fscores = new List<double>();

                    //Real image calculations
                    double GError = 0;
                    double RealPercCorrect = 0;
                    Critic.Calculate(realsamples);
                    for (int j = 0; j < BatchSize; j++) 
                    { 
                        //The score is the value of the output (last) neuron of the critic
                        rscores.Add(Critic.Layers[Critic.NumLayers - 1].Values[j][0]);
                        AvgRealScore += rscores[j];
                        //Add the squared error
                        overallscore += Math.Pow(1d - Critic.Layers[Critic.NumLayers - 1].Values[j][0], 2);
                        GError += Math.Pow(-Critic.Layers[Critic.NumLayers - 1].Values[j][0], 2);
                        //Add whether it was correct or not to the total
                        RealPercCorrect += Critic.Layers[Critic.NumLayers - 1].Values[j][0] > 0 ? 1d : 0d;
                    }
                    AvgRealScore /= BatchSize;
                    RealPercCorrect /= BatchSize;
                    //Loss on real images = how accurate the critic is
                    Critic.CalcGradients(realsamples, null, RealPercCorrect, true);

                    //Fake image calculations
                    double FakePercIncorrect = 0;
                    Critic.Calculate(fakesamples);
                    for (int j = 0; j < BatchSize; j++)
                    {
                        //The score is the value of the output (last) neuron of the critic
                        fscores.Add(Critic.Layers[Critic.NumLayers - 1].Values[j][0]);
                        AvgFakeScore += fscores[j];
                        //Add the squared error
                        overallscore += Math.Pow(-Critic.Layers[Critic.NumLayers - 1].Values[j][0], 2);
                        GError += Math.Pow(1d - Critic.Layers[Critic.NumLayers - 1].Values[j][0], 2);
                        //Add whether it was correct or not to the total
                        FakePercIncorrect += Critic.Layers[Critic.NumLayers - 1].Values[j][0] > 0 ? 1d : 0d;
                    }
                    AvgFakeScore /= BatchSize;
                    FakePercIncorrect /= BatchSize;
                    //Wasserstein loss = real % correct - fake % correct
                    Critic.CalcGradients(fakesamples, null, RealPercCorrect - (1 - FakePercIncorrect), true);


                    //Update weights and biases
                    Critic.Update(gradientnorm);

                    //Report values to the front end
                    if (Clear) 
                    { 
                        Critic.Trials = 0;
                        Generator.Trials = 0;
                        Clear = false;
                    }
                    overallscore = Math.Sqrt(overallscore / (2 * BatchSize));
                    GError = Math.Sqrt(GError / BatchSize);
                    
                    //Update errors and % correct values
                    Critic.Error = (Critic.Error * ((Critic.Trials) / (Critic.Trials + 1d))) + (overallscore * (1d / (Critic.Trials + 1d)));
                    Critic.PercCorrect = (Critic.PercCorrect * ((Critic.Trials) / (Critic.Trials + 1d))) + (RealPercCorrect * (1d / (Critic.Trials + 1d)));
                    Generator.Error = (Generator.Error * ((Generator.Trials) / (Generator.Trials + 1d))) + (GError * (1d / (Generator.Trials + 1)));
                    Generator.PercCorrect = (Generator.PercCorrect * ((Generator.Trials) / (Generator.Trials + 1d))) + (FakePercIncorrect * (1d / (Generator.Trials + 1d)));
                    //Iterate trial count
                    Critic.Trials++;
                    Generator.Trials++;
                }
                //Adjust loss for batch size and critic to generator ratio
                totalrealmean /= ctg;
                totalrealstddev /= ctg;

                //Generate samples
                List<double[]> testlatents = new List<double[]>();
                for (int i = 0; i < BatchSize; i++) { testlatents.Add(Maths.RandomGaussian(r, LatentSize)); }
                var tests = Generator.GenerateSamples(testlatents);

                //Criticize generated samples
                Critic.Calculate(tests);

                //Compute generator's error on the critic's scores
                double Score = 0;
                for (int j = 0; j < BatchSize; j++)
                {
                    Score += Critic.Layers[Critic.NumLayers - 1].Values[j][0] > 0 ? 1 : 0;
                }

                //Backprop through the critic to the generator
                Critic.CalcGradients(tests, null, Score, false);
                Generator.CalcGradients(testlatents, Critic.Layers[0], Score, true);

                //Update the generator's weights and biases
                Generator.Update(gradientnorm);

                //Update image (if applicable)
                if (formupdateiterator >= imgspeed)
                {
                    //Code that converts normalized generator outputs into an image
                    //Changes distribution of output values to 0-255 (brightness)
                    var values = Form1.Rescale(Maths.Convert(tests[0]));
                    var image = new int[28, 28];
                    //Convert values to a 2d int array
                    for (int i = 0; i < 28; i++)
                    {
                        for (int ii = 0; ii < 28; ii++)
                        {
                            image[ii, i] = (int)values[i, ii];
                        }
                    }
                    //Report values and image to the front end
                    activeform.Invoke((Action)delegate
                    {
                        activeform.image = image; 
                        activeform.CScore = Critic.Error.ToString();
                        activeform.CPerc = Critic.PercCorrect.ToString();
                        activeform.GScore = Generator.Error.ToString();
                        activeform.GPerc = Generator.PercCorrect.ToString();
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
                activeform.CPerc = null;
                activeform.GScore = null;
                activeform.GPerc = null;
            });
        }
        /// <summary>
        /// Computes the values and zvalues of each layer using either the input image or previous layer's values
        /// </summary>
        /// <param name="inputs">The input image</param>
        public void Calculate(List<double[]> inputs)
        {
            Calculate(Layers[0], inputs, ResidualLayers[0], false);
            for (int i = 1; i < NumLayers; i++)
            {
                Calculate(Layers[i], Layers[i - 1].Values, ResidualLayers[i], i == NumLayers - 1);
            }
            Residuals = null;
        }
        /// <summary>
        /// Processes residuals and calculates values for a specific layer
        /// </summary>
        /// <param name="layer">The layer being computed</param>
        /// <param name="inputs">The input of the former layer</param>
        /// <param name="isResidual">Whether the layer is a residual</param>
        /// <param name="isoutput">Whether the layer is the output of the network</param>
        public void Calculate(Layer layer, List<double[]> inputs, bool isResidual, bool isoutput)
        {
            //If a sum layer
            if (layer is SumLayer)
            {
                //Sum with an all-0 matrix to preserve the input if none exists
                if (Residuals is null)
                {
                    Residuals = new List<double[]>();
                    for (int lol = 0; lol < inputs.Count; lol++)
                    {
                        Residuals.Add(new double[layer.InputLength]);
                    }
                }
                //Otherwise sum with the most recent residual layer
                (layer as SumLayer).Calculate(Residuals, inputs);
            }
            //Otherwise calculate the next set of values
            else { layer.Calculate(inputs, isoutput); }

            //If this layer is a residual, its values are the next residual
            if (isResidual)
            { 
                Residuals = layer.Values;
            }
        }
        public void CalcGradients(List<double[]> inputs, Layer output, double correct, bool calcgradients)
        {
            //Output layer
            Layers[NumLayers - 1].Backprop(Layers[NumLayers - 2].Values, output, correct, calcgradients);
            //Middle layers
            for (int i = NumLayers - 2; i >= 1; i--)
            {
                Layers[i].Backprop(Layers[i - 1].Values, Layers[i + 1], -99, calcgradients);
                if (Layers[i] is SumLayer)
                {
                    int j = i;
                    while (!ResidualLayers[j] && j >= -1) { j--; }
                    if (j == -1) { throw new Exception("Invalid ratio of residual to sum layers"); }
                    Layers[j].Backprop(Layers[j - 1].Values, Layers[i], -99, calcgradients);
                }
            }
            //Input layer
            Layers[0].Backprop(inputs, Layers[1], -99, calcgradients);
        }
        /// <summary>
        /// Updates the NN's layer's weights after a full batch of gradient descent
        /// </summary>
        /// <param name="m">Batch size</param>
        /// <param name="a">Learning rate</param>
        /// <param name="c">Clipping parameter</param>
        /// <param name="rmsd">RMSProp decay parameter</param>
        public void Update(bool batchnorm)
        {
            for (int i = 0; i < NumLayers; i++)
            {
                //These layer types don't have weights/biases so don't update them
                if (Layers[i] is SumLayer || Layers[i] is PoolingLayer) { continue; }
                //Update the weights of [i] layer
                Layers[i].Descend(batchnorm);
            }
        }
        /// <summary>
        /// Computes the generator's values and returns them
        /// </summary>
        /// <param name="latentspaces">The random noise input of the network</param>
        /// <returns></returns>
        List<double[]> GenerateSamples(List<double[]> latentspaces)
        {
            Calculate(latentspaces);
            return Layers[NumLayers - 1].Values;
        }
        List<double[]> GenerateNoisyImages(Random r, int latentsize, int num)
        {
            var output = new List<double[]>();
            for (int i = 0; i < latentsize; i++)
            {
                var img = IO.FindNextNumber(num);
                var latentspace = Maths.RandomGaussian(r, latentsize);
                output.Add(new double[latentsize]);
                for (int j = 0; j < latentsize; j++)
                {
                    output[i][j] = img[j] * latentspace[j];
                }
            }
            return output;
        }
    }
}
