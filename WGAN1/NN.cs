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
        public List<bool> TanhLayers { get; set; }
        public List<bool> ResidualLayers { get; set; }
        public List<bool> BatchNormLayers { get; set; }
        List<double[]> Residuals { get; set; }
        public static double LearningRate = 0.00005;
        public static double RMSDecay = .9;
        public static bool UseRMSProp = true;
        public static double Infinitesimal = 1E-20;
        public static bool UseClipping = false;
        public static double ClipParameter = 10;
        public double BatchSize { get; set; }
        public static bool Training = false;
        public static bool Clear = false;
        public static bool Save = true;
        public int OutputLength { get; set; }
        int Trials = 0;
        public double Error = 0;
        public double PercCorrect = 0;

        /// <summary>
        /// Generates a new NN with the specified parameters (using LeCun initialization)
        /// </summary>
        /// <param name="l">Number of layers in the network</param>
        /// <param name="wcs">Number of weights/biases in the network</param>
        public NN Init(List<Layer> layers, List<bool> Tanhs, List<bool> residuals, List<bool> batchnorms)
        {
            Layers = layers;
            NumLayers = Layers.Count;
            TanhLayers = Tanhs;
            ResidualLayers = residuals;
            BatchNormLayers = batchnorms;
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Init(i == NumLayers - 1);
                if (TanhLayers[i]) { Layers[i].UsesTanh = true; }
                else { Layers[i].UsesTanh = false; }
            }
            OutputLength = Layers[NumLayers - 1].OutputLength;
            return this;
        }
        public static void TestTrain (NN Critic, int batchsize, int imgspeed, Form1 activeform)
        {
            int formupdateiterator = 0;
            while (Training)
            {
                double mean1 = 0, mean8 = 0;
                double stddev1 = 0, stddev8 = 0;
                double score = 0;
                double perccorrect = 0;
                List<double[]> list8 = new List<double[]>();
                List<double[]> list1 = new List<double[]>();
                for (int i = 0; i < batchsize; i++)
                {
                    list1.Add(IO.FindNextNumber(1));
                    var mean = Maths.CalcMean(list1[i]);
                    mean1 += mean;
                    stddev1 += Maths.CalcStdDev(list1[i], mean);
                    list8.Add(IO.FindNextNumber(8));
                    mean = Maths.CalcMean(list8[i]);
                    mean8 += mean;
                    stddev8 += Maths.CalcStdDev(list8[i], mean);
                }
                mean8 /= batchsize; mean1 /= batchsize;
                stddev1 /= batchsize; stddev8 /= batchsize;

                list1 = Maths.BatchNormalize(list1, mean1, stddev1);
                list8 = Maths.BatchNormalize(list8, mean8, stddev8);

                Critic.Calculate(list1);
                foreach (double[] d in Critic.Layers[Critic.NumLayers - 1].ZVals) 
                {
                    score += (1 - d[0]) * (1 - d[0]);
                    score += (-d[1]) * (-d[1]);
                    perccorrect += d[0] > d[1] ? 1 : 0;
                }
                Critic.CalcGradients(list1, null, 0, true);

                Critic.Calculate(list8);
                foreach (double[] d in Critic.Layers[Critic.NumLayers - 1].ZVals)
                {
                    score += (1 - d[1]) * (1 - d[1]);
                    score += (-d[0]) * (-d[0]);
                    perccorrect += d[0] > d[1] ? 0 : 1;
                }
                score /= (2 * batchsize);
                perccorrect /= (2 * batchsize);

                score = Math.Sqrt(score);
                Critic.CalcGradients(list8, null, 1, true);

                Critic.Update(batchsize, true);

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
                    var values = Form1.Rescale(Maths.Convert(list1[0]));
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
               
                for (int i = 0; i < ctg; i++)
                {
                    //Batch norm stuff
                    double realmean = 0;
                    double realstddev = 0;

                    double AvgFakeScore = 0;
                    double AvgRealScore = 0;

                    //Generate samples
                    var realsamples = new List<double[]>();
                    var latentspaces = new List<double[]>();
                    for (int ii = 0; ii < m; ii++)
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
                    realmean /= m; totalrealmean += realmean; 
                    realstddev /= m; totalrealstddev += realstddev;

                    //Batchnorm the samples
                    realsamples = Maths.BatchNormalize(realsamples, realmean, realstddev);
                    var fakesamples = Generator.GenerateSamples(latentspaces);
                    //var fakesamples = Maths.BatchNormalize(Generator.GenerateSamples(latentspaces));

                    double overallscore = 0;
                    //Critic's scores of each type of sample
                    List<double> rscores = new List<double>();
                    List<double> fscores = new List<double>();
                    //Compute values and loss
                    //Wasserstein loss = Avg(CScore(real) - CScore(fake))

                    //Reals

                    //Real image calculations
                    double GError = 0;
                    Critic.Calculate(realsamples);
                    for (int j = 0; j < m; j++) 
                    { 
                        rscores.Add(Critic.Layers[Critic.NumLayers - 1].ZVals[j][0]);
                        overallscore += Math.Pow(1d - Critic.Layers[Critic.NumLayers - 1].ZVals[j][1], 2);
                        overallscore += Math.Pow(-Critic.Layers[Critic.NumLayers - 1].ZVals[j][0], 2);
                        GError += Math.Pow(-Critic.Layers[Critic.NumLayers - 1].ZVals[j][0], 2);
                        AvgRealScore += rscores[j];
                    }
                    AvgRealScore /= m;
                    Critic.CalcGradients(realsamples, null, 1, true);
                    //Fakes

                    //Fake image calculations
                    Critic.Calculate(fakesamples);
                    for (int j = 0; j < m; j++)
                    {
                        fscores.Add(Critic.Layers[Critic.NumLayers - 1].ZVals[j][0]);
                        overallscore += Math.Pow(1d - Critic.Layers[Critic.NumLayers - 1].ZVals[j][0], 2);
                        overallscore += Math.Pow(-Critic.Layers[Critic.NumLayers - 1].ZVals[j][1], 2);
                        GError += Math.Pow(1d - Critic.Layers[Critic.NumLayers - 1].ZVals[j][0], 2);
                        AvgFakeScore += fscores[j];
                    }
                    AvgFakeScore /= m;
                    Critic.CalcGradients(fakesamples, null, 0, true);

                    //Critic's Wassertsein loss
                    double CWLoss = AvgRealScore - AvgFakeScore;
                    //Update
                    Critic.Update(m, gradientnorm);

                    //Report values to the front end
                    if (Clear) 
                    { 
                        Critic.Trials = 0; Critic.Error = 0; Critic.PercCorrect = 0;
                        Generator.Trials = 0; Generator.Error = 0; Generator.PercCorrect = 0; Clear = false;
                    }
                    overallscore = Math.Sqrt(overallscore / (2 * m));
                    GError = Math.Sqrt(GError / m);
                    Critic.Trials++;
                    Generator.Trials++;
                    Critic.Error = (Critic.Error * ((Critic.Trials) / (Critic.Trials + 1d))) + (overallscore * (1d / (Critic.Trials)));
                    Generator.Error = (Generator.Error * ((Generator.Trials) / (Generator.Trials + 1d))) + (GError * (1d / (Generator.Trials)));
                }
                //Adjust loss for batch size and critic to generator ratio
                totalrealmean /= ctg;
                totalrealstddev /= ctg;
                //Train generator
                List<double[]> testlatents = new List<double[]>();
                for (int i = 0; i < m; i++) { testlatents.Add(Maths.RandomGaussian(r, LatentSize)); }
                var tests = Generator.GenerateSamples(testlatents);
                //var tests = Maths.BatchNormalize(Generator.GenerateSamples(testlatents));
                Critic.Calculate(tests);
                //The generator's wasserstein loss
                double GWLoss = 0;
                for (int j = 0; j < m; j++)
                {
                    GWLoss += Critic.Layers[Critic.NumLayers - 1].ZVals[j][0];
                }
                Critic.CalcGradients(tests, null, 1, false);
                //Backprop through the critic to the generator
                Generator.CalcGradients(tests, Critic.Layers[0], 1, true);
                //Update
                Generator.Update(m, gradientnorm);

                //Update image (if applicable)
                if (formupdateiterator >= imgspeed)
                {
                    //Code that converts normalized generator outputs into an image
                    //Changes distribution of output values to 0-255 (brightness)
                    totalrealmean /= ctg; totalrealstddev /= ctg;
                    var values = Maths.Rescale(tests[0], totalrealmean, totalrealstddev);
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
        public void Calculate(List<double[]> inputs)
        {
            Calculate(Layers[0], inputs, TanhLayers[0], ResidualLayers[0], BatchNormLayers[0], false);
            for (int i = 1; i < NumLayers; i++)
            {
                Calculate(Layers[i], Layers[i - 1].ZVals, TanhLayers[i], ResidualLayers[i], BatchNormLayers[i], i == NumLayers - 1);
            }
            Residuals = null;
        }
        public void Calculate(Layer layer, List<double[]> inputs, bool tanh, bool isResidual, bool batchnorm, bool isoutput)
        {
            if (batchnorm)
            {
                //Calculate mean and stddev of the inputs
                double mean = 0, stddev = 0;
                foreach (double[] d in inputs)
                {
                    double inputmean = Maths.CalcMean(d);
                    mean += inputmean;
                    stddev += Maths.CalcStdDev(d, inputmean);
                }
                mean /= inputs.Count; stddev /= inputs.Count;
                for (int i = 0; i < inputs.Count; i++)
                {
                    inputs[i] = Maths.Normalize(inputs[i], mean, stddev);
                }
            }
            if (tanh)
            {
                //Apply tanh and batchnorm to the inputs
                for (int i = 0; i < inputs.Count; i++)
                {
                    inputs[i] = Maths.Tanh(inputs[i]);
                }
            }

            //Calculate the next set of values
            if (layer is SumLayer)
            {
                //Just sum with an all-0 matrix to preserve the input lol
                if (Residuals is null)
                {
                    Residuals = new List<double[]>();
                    for (int lol = 0; lol < inputs.Count; lol++)
                    {
                        Residuals.Add(new double[layer.InputLength]);
                    }
                }
                    (layer as SumLayer).Calculate(Residuals, inputs);
            }
            else { layer.Calculate(inputs, isoutput); }

            if (isResidual)
            { 
                Residuals = layer.ZVals;
            }
        }
        public void CalcGradients(List<double[]> inputs, Layer output, int correct, bool calcgradients)
        {
            //Output layer
            Layers[NumLayers - 1].Backprop(Layers[NumLayers - 2].ZVals, output, correct, calcgradients);
            //Middle layers
            for (int i = NumLayers - 2; i >= 1; i--)
            {
                Layers[i].Backprop(Layers[i - 1].ZVals, Layers[i + 1], -99, calcgradients);
            }
            //Input layer
            Layers[0].Backprop(inputs, Layers[1], -99, calcgradients);

            //Add residual errors

            //The process is to find the most recent sum layer, then
            //backprop its errors to the corresponding (aka most recent) residual layer
            int j = NumLayers - 1;
            Layer most_recent_sumlayer = null;
            do
            {
                //Add errors to the layer whose values were taken
                if (ResidualLayers[j])
                {
                    List<double[]> input = inputs;
                    if (j != 0) { input = Layers[j - 1].ZVals; }
                    Layers[j].Backprop(input, most_recent_sumlayer, -99, calcgradients);
                    most_recent_sumlayer = null;
                }
                if (Layers[j] is SumLayer) { most_recent_sumlayer = Layers[j]; }
                j--;
            }
            while (j >= 0);
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
                if (Layers[i] is SumLayer || Layers[i] is PoolingLayer) { continue; }
                 
                 Layers[i].Descend(m, batchnorm);
            }
        }
        List<double[]> GenerateSamples(List<double[]> latentspaces)
        {
            Calculate(latentspaces);
            return Layers[NumLayers - 1].ZVals;
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
