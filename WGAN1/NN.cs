using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;
using System.Windows.Markup;

namespace WGAN1
{
    class NN
    {
        public int NumLayers { get; set; }
        public List<iLayer> Layers { get; set; }
        public List<bool> TanhLayers { get; set; }
        public List<bool> ResidualLayers { get; set; }
        public List<bool> BatchNormLayers { get; set; }
        List<double[]> Residuals { get; set; }
        List<List<double[]>> Values { get; set; }
        int ResidualIndex { get; set; }
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
        int Trials = 0;
        public double Error = 0;

        /// <summary>
        /// Generates a new NN with the specified parameters (using LeCun initialization)
        /// </summary>
        /// <param name="l">Number of layers in the network</param>
        /// <param name="wcs">Number of weights/biases in the network</param>
        public NN Init(List<iLayer> layers, List<bool> Tanhs, List<bool> residuals, List<bool> batchnorms)
        {
            Layers = layers;
            NumLayers = Layers.Count;
            TanhLayers = Tanhs;
            ResidualLayers = residuals;
            BatchNormLayers = batchnorms;
            for (int i = 0; i < NumLayers; i++)
            {
                Layers[i].Init(i == NumLayers -1);
                if (TanhLayers[i]) { Layers[i].UsesTanh = true; }
                else { Layers[i].UsesTanh = false; }
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

                    //Batchnorm the real samples
                    for (int j = 0; j < m; j++)
                    {
                        realsamples[j] = Maths.Normalize(realsamples[j], realmean, realstddev);
                    }
                    var fakesamples = Generator.GenerateSamples(latentspaces);

                    double overallscore = 0;
                    //Critic's scores of each type of sample
                    List<double> rscores = new List<double>();
                    List<double> fscores = new List<double>();
                    //Compute values and loss
                    //Wasserstein loss = Avg(CScore(real) - CScore(fake))

                    //Reals

                    //Real image calculations
                    Critic.Calculate(realsamples);
                    for (int j = 0; j < m; j++) 
                    { 
                        rscores.Add(Critic.Values[Critic.NumLayers - 1][j][0]);
                        CWLoss += rscores[j];
                    }
                    CWLoss /= m;
                    //Backprop
                    Critic.CalcGradients(realsamples, null, CWLoss, true);
                    for (int j = 0; j < m; j++)
                    {
                        overallscore += Critic.Values[Critic.NumLayers - 1][j][0] > 0 ? 1 : 0;
                    }

                    //Fakes
                    
                    //Fake image calculations
                    Critic.Calculate(fakesamples);
                    CWLoss = 0;
                    for (int j = 0; j < m; j++)
                    {
                        fscores.Add(Critic.Values[Critic.NumLayers - 1][j][0]);
                        CWLoss += fscores[j];
                        AvgFakeScore += fscores[j];
                    }
                    CWLoss /= m;
                    Critic.CalcGradients(fakesamples, null, -CWLoss, true);

                    //Calculate the generator's error
                    double GError = 0;
                    for (int j = 0; j < m; j++)
                    {
                        overallscore += Critic.Values[Critic.NumLayers - 1][j][0] < 0 ? 1 : 0;
                        GError += Critic.Values[Critic.NumLayers - 1][j][0];
                    }                  

                    //Update
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
                totalrealmean /= ctg;
                totalrealstddev /= ctg;
                //Train generator
                List<double[]> testlatents = new List<double[]>();
                for (int i = 0; i < m; i++) { testlatents.Add(Maths.RandomGaussian(r, LatentSize)); }
                var tests = Generator.GenerateSamples(testlatents);
                Critic.Calculate(tests);
                Critic.CalcGradients(tests, null, -AvgFakeScore, false);
                //Backprop through the critic to the generator
                Generator.CalcGradients(testlatents, Critic.Layers[0], -AvgFakeScore, true);
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
            Values = new List<List<double[]>>();
            Values.Add(Calculate(Layers[0], inputs, TanhLayers[0], ResidualLayers[0], BatchNormLayers[0], false));
            for (int i = 1; i < NumLayers; i++)
            {
                Values.Add(Calculate(Layers[i], Values[i - 1], TanhLayers[i], ResidualLayers[i], BatchNormLayers[i], i == NumLayers - 1));
                //if (Layers[i] is SumLayer)
                //{
                //    for (int j = 0; j < Values[i].Count; j++)
                //    {
                //        (Layers[i] as SumLayer).Calculate(Values[i][j], Values[ResidualIndex][j]);
                //        Values[i].Add(Layers[i].ZVals);
                //    }
                //}
                //else
                //{
                //    Values.Add(Calculate(Layers[i], Values[i - 1], ResidualLayers[i], BatchNormLayers[i], i == NumLayers - 1));
                //}
            }
        }
        public List<double[]> Calculate(iLayer layer, List<double[]> inputs, bool tanh, bool isResidual, bool batchnorm, bool isoutput)
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
            List<double[]> outputs = new List<double[]>();
            for (int i = 0; i < inputs.Count; i++)
            {
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
                    (layer as SumLayer).Calculate(Residuals[i], inputs[i]);
                }
                else { layer.Calculate(inputs[i], isoutput); }
                outputs.Add(layer.ZVals);
            }
            if (isResidual)
            { 
                Residuals = outputs;
            }
            return outputs;
        }
        public void CalcGradients(List<double[]> inputs, iLayer output, double correct, bool calcgradients)
        {
            //Reset errors
            Layers[NumLayers - 1].Errors = new double[Layers[NumLayers - 1].Length];
            for (int i = 0; i < NumLayers - 1; i++)
            {
                Layers[i].Errors = new double[Layers[i + 1].InputLength];
            }
            //Output layer
            for (int i = 0; i < Values[NumLayers - 1].Count; i++) 
            {
                Layers[NumLayers - 1].Backprop(Values[NumLayers - 2][i], output, Values[NumLayers - 1][i], correct, calcgradients);
            }
            //Middle layers
            for (int i = NumLayers - 2; i >= 1; i--)
            {
                for (int ii  = 0; ii < Values[i - 1].Count; ii++)
                {
                    Layers[i].Backprop(Values[i - 1][ii], Layers[i + 1], null, -99, calcgradients);
                }
            }
            //Input layer
            for (int i = 0; i < inputs.Count; i++)
            {
                Layers[0].Backprop(inputs[i], Layers[1], null, -99, calcgradients);
            }

            //Add residual errors

            //The process is to find the most recent sum layer, then
            //backprop its errors to the corresponding (aka most recent) residual layer
            int j = NumLayers - 1;
            do
            {
                iLayer most_recent_sumlayer = null;
                if (Layers[j] is SumLayer) { most_recent_sumlayer = Layers[j]; }
                //Add errors to the layer whose values were taken
                if (ResidualLayers[j])
                {
                    List<double[]> input = inputs;
                    if (j != 0) { input = Values[j - 1]; }
                    for (int i = 0; i < input.Count; i++)
                    {
                        Layers[j].Backprop(input[i], most_recent_sumlayer, null, -99, calcgradients);
                    }
                }
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
                if (Layers[i] is SumLayer) { continue; }
                Layers[i].Descend(m, batchnorm);
            }
        }
        List<double[]> GenerateSamples(List<double[]> latentspaces)
        {
            Calculate(latentspaces);
            return Values[NumLayers - 1];
        }
    }
}
