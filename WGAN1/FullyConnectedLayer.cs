using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class FullyConnectedLayer : iLayer
    {
        public int Length { get; set; }
        public int InputLength { get; set; }
        public double[] Values { get; set; }
        public double[] ZVals { get; set; }
        public double[] Errors { get; set; }
        public double[,] Weights { get; set; }
        public double[] Biases { get; set; }
        double[,] WRMSGrad { get; set; }
        double[,] WeightGradient { get; set; }
        double[] BRMSGrad { get; set; }
        double[] BiasGradient { get; set; }
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
        public void Descend(int batchsize)
        {
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Normal gradient descent update
                    double gradient = WeightGradient[i, ii] * (-2d / batchsize);
                    double update = NN.LearningRate * gradient;
                    //Root mean square propegation
                    if (NN.UseRMSProp)
                    {
                        WRMSGrad[i, ii] = (WRMSGrad[i, ii] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (gradient * gradient));
                        update = (NN.LearningRate / Math.Sqrt(WRMSGrad[i, ii])) * gradient;
                    }
                    //Update weight and average
                    Weights[i, ii] -= update;
                    AvgGradient -= update;
                    //Gradient clipping
                    if (NN.UseClipping)
                    {
                        if (Weights[i, ii] > NN.ClipParameter) { Weights[i, ii] = NN.ClipParameter; }
                        if (Weights[i, ii] < -NN.ClipParameter) { Weights[i, ii] = -NN.ClipParameter; }
                    }
                }
                //Normal gradient descent update
                double bgradient = BiasGradient[i] * (-2d / batchsize);
                double bupdate = NN.LearningRate * bgradient;
                //Root mean square propegation
                if (NN.UseRMSProp)
                {
                    BRMSGrad[i] = (BRMSGrad[i] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (bgradient * bgradient));
                    bupdate = (NN.LearningRate / Math.Sqrt(BRMSGrad[i])) * bgradient;
                }
                //Gradient clipping
                if (NN.UseClipping)
                {
                    if (bupdate > NN.ClipParameter) { bupdate = NN.ClipParameter; }
                    if (bupdate < -NN.ClipParameter) { bupdate = -NN.ClipParameter; }
                }
                //Update bias
                Biases[i] -= bupdate;
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
        public void Backprop(double[] input, iLayer outputlayer, bool isoutput, double correct, bool calcgradients)
        {
            //Calculate error
            if (isoutput)
            {
                Errors = new double[Length];
                for (int i = 0; i < Length; i++)
                {
                    Errors[i] = 2d * (correct - Values[i]);
                }
            }
            else
            {
                if (outputlayer is FullyConnectedLayer)
                {
                    var FCLOutput = outputlayer as FullyConnectedLayer;
                    Errors = new double[Length];
                    for (int k = 0; k < FCLOutput.Length; k++)
                    {
                        for (int j = 0; j < Length; j++)
                        {
                            Errors[j] += FCLOutput.Weights[k, j] * Maths.TanhDerriv(outputlayer.ZVals[k]) * FCLOutput.Errors[k];
                        }
                    }
                }
                if (outputlayer is ConvolutionLayer)
                {
                    var CLOutput = outputlayer as ConvolutionLayer;
                    Errors = Maths.Convert(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)));
                }
            }
            if (calcgradients)
            {
                //Calculate gradients
                for (int i = 0; i < Length; i++)
                {
                    for (int ii = 0; ii < InputLength; ii++)
                    {
                        //Weight gradients
                        WeightGradient[i, ii] = input[ii] * Maths.TanhDerriv(ZVals[i]) * Errors[i];
                    }
                    if (isoutput) { continue; }
                    //Bias gradients
                    BiasGradient[i] = Maths.TanhDerriv(ZVals[i]) * Errors[i];
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
            if (!output) { Values = Maths.Tanh(vals); }
            else { Values = vals; }
        }
        public void Calculate(double[,] input, bool output)
        {
            Calculate(Maths.Convert(input), output);
        }
    }
}
