using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class FullyConnectedLayer : iLayer
    {
        public int Length { get; set; }
        public int InputLength { get; set; }
        public bool UsesTanh { get; set; }
        public double[] ZVals { get; set; }
        public double[] Errors { get; set; }
        //Weights
        public double[,] Weights { get; set; }
        double[,] WRMSGrad { get; set; }
        double[,] WeightGradient { get; set; }
        double[,] WUpdates { get; set; }
        //Biases
        public double[] Biases { get; set; }
        double[] BRMSGrad { get; set; }
        double[] BiasGradient { get; set; }
        double[] BUpdates { get; set; }

        public double AvgGradient { get; set; }
        public FullyConnectedLayer(int l, int il)
        {
            Length = l; InputLength = il;
            WeightGradient = new double[l, il];
            WRMSGrad = new double[l, il];
            BiasGradient = new double[l];
            BRMSGrad = new double[l];
            Weights = new double[l, il];
            Biases = new double[l];
        }
        public iLayer Init(bool isoutput)
        {
            var r = new Random();
            //All layers have weights
            Weights = new double[Length, InputLength];
            //Output layer has no biases
            if (!isoutput) { Biases = new double[Length]; }
            //Initialize weights (and biases to zero)
            for (int j = 0; j < Length; j++)
            {
                for (int jj = 0; jj < InputLength; jj++)
                {
                    Weights[j, jj] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (InputLength * InputLength));
                }
            }
            return this;
        }
        /// <summary>
        /// Applies the gradients to the weights as a batch
        /// </summary>
        /// <param name="batchsize">The number of trials run per cycle</param>
        /// <param name="clipparameter">What the max/min </param>
        /// <param name="RMSDecay">How quickly the RMS gradients decay</param>
        public void Descend(int batchsize, bool batchnorm)
        {
            //Calculate gradients
            WUpdates = new double[Length, InputLength];
            BUpdates = new double[Length];
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Normal gradient descent update
                    WUpdates[i, ii] = NN.LearningRate * WeightGradient[i, ii] * (-2d / batchsize);
                    //Root mean square propegation
                    if (NN.UseRMSProp)
                    {
                        WRMSGrad[i, ii] = (WRMSGrad[i, ii] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (WUpdates[i, ii] * WUpdates[i, ii])) + NN.Infinitesimal;
                        WUpdates[i, ii] = (NN.LearningRate / Math.Sqrt(WRMSGrad[i, ii])) * WUpdates[i, ii];
                    }
                }
                //Normal gradient descent update
                BUpdates[i] = BiasGradient[i] * (-2d / batchsize);
                //Root mean square propegation
                if (NN.UseRMSProp)
                {
                    BRMSGrad[i] = (BRMSGrad[i] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (BUpdates[i] * BUpdates[i]));
                    BUpdates[i] = (NN.LearningRate / Math.Sqrt(BRMSGrad[i])) * BUpdates[i];
                }
            }
            //Gradient normalization
            if (batchnorm) 
            { 
                WUpdates = Maths.Scale(NN.LearningRate, Maths.Normalize(WUpdates)); 
                BUpdates = Maths.Scale(NN.LearningRate, Maths.Normalize(BUpdates)); 
            }
            //Apply updates
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Update weight and average
                    Weights[i, ii] -= WUpdates[i, ii];
                    AvgGradient -= WUpdates[i, ii];
                    //Gradient clipping
                    if (NN.UseClipping)
                    {
                        if (Weights[i, ii] > NN.ClipParameter) { Weights[i, ii] = NN.ClipParameter; }
                        if (Weights[i, ii] < -NN.ClipParameter) { Weights[i, ii] = -NN.ClipParameter; }
                    }
                }
                Biases[i] -= NN.LearningRate * BUpdates[i];
                //Gradient clipping
                if (NN.UseClipping)
                {
                    if (Biases[i] > NN.ClipParameter) { Biases[i] = NN.ClipParameter; }
                    if (Biases[i] < -NN.ClipParameter) { Biases[i] = -NN.ClipParameter; }
                }
            }
            //Reset gradients (but not RMS gradients)
            WeightGradient = new double[Length, InputLength];
            BiasGradient = new double[Length];
        }
        /// <summary>
        /// Descent for other layers
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="isoutput">Whether the layer is the output layer</param>
        /// <summary>
        /// Backpropegation of error and calcluation of gradients
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="isoutput">Whether the layer is the output layer</param>
        public void Backprop(double[] input, iLayer outputlayer, double[] outputvals, double correct, bool calcgradients)
        {
            //Calculate error
            if (!(outputvals is null))
            {
                for (int i = 0; i < Length; i++)
                {
                    Errors[i] = 2d * (outputvals[i] - correct);
                }
            }
            else
            {
                if (outputlayer is SumLayer)
                {
                    //Errors with respect to the output of the convolution
                    //dl/do
                    for (int k = 0; k < outputlayer.Length; k++)
                    {
                        for (int j = 0; j < outputlayer.InputLength; j++)
                        {
                            double zvalderriv = ZVals[j];
                            if (outputlayer.UsesTanh) { zvalderriv = Maths.TanhDerriv(zvalderriv); }
                            Errors[j] += zvalderriv * outputlayer.Errors[k];
                        }
                    }
                }
                if (outputlayer is FullyConnectedLayer)
                {
                    var FCLOutput = outputlayer as FullyConnectedLayer;
                    for (int k = 0; k < FCLOutput.Length; k++)
                    {
                        for (int j = 0; j < Length; j++)
                        {
                            double zvalderriv = outputlayer.ZVals[k];
                            if (outputlayer.UsesTanh) { zvalderriv = Maths.TanhDerriv(zvalderriv); }
                            Errors[j] += FCLOutput.Weights[k, j] * zvalderriv * FCLOutput.Errors[k];
                        }
                    }
                }
                if (outputlayer is ConvolutionLayer)
                {
                    var CLOutput = outputlayer as ConvolutionLayer;
                    if ((outputlayer as ConvolutionLayer).DownOrUp) { Errors = Maths.Convert(CLOutput.UnPad(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)))); }
                    else { Errors = Maths.Convert(CLOutput.UnPad(CLOutput.Convolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)))); }
                    //Errors = Maths.Convert(CLOutput.UnPad(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors))));
                }
            }
            if (calcgradients)
            {
                //Calculate gradients
                for (int i = 0; i < Length; i++)
                {
                    double zvalderriv = ZVals[i];
                    if (UsesTanh) { zvalderriv = Maths.TanhDerriv(ZVals[i]); }

                    for (int ii = 0; ii < InputLength; ii++)
                    {
                        //Weight gradients
                        WeightGradient[i, ii] = -1 * input[ii] * zvalderriv * Errors[i];
                    }
                    if (!(outputvals is null)) { continue; }
                    //Bias gradients
                    BiasGradient[i] = -1 * zvalderriv * Errors[i];
                }
            }
        }
        public void Calculate(double[] input, bool output)
        {
            var vals = new double[Length];
            for (int k = 0; k < Length; k++)
            {
                for (int j = 0; j < InputLength; j++)
                {
                    vals[k] += Weights[k, j] * input[j];
                }
                if (!output)
                {
                    vals[k] += Biases[k];
                }
            }
            ZVals = vals;
        }
        public void Calculate(double[,] input, bool output)
        {
            Calculate(Maths.Convert(input), output);
        }
    }
}
