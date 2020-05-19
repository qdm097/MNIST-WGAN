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
        /// <param name="learningrate">The rate of learning</param>
        /// <param name="clipparameter">What the max/min </param>
        /// <param name="RMSDecay">How quickly the RMS gradients decay</param>
        public void Descend(int batchsize, double learningrate, double clipparameter, double RMSDecay)
        {
            AvgGradient = 0;
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Normal gradient descent update
                    double gradient = WeightGradient[i, ii] * (-2d / batchsize);
                    //Definition of RMSProp
                    WRMSGrad[i, ii] = (WRMSGrad[i, ii] * RMSDecay) + ((1 - RMSDecay) * (gradient * gradient));
                    double update = (learningrate / Math.Sqrt(WRMSGrad[i, ii])) * gradient;
                    //Gradient clipping
                    if (update > clipparameter) { update = clipparameter; }
                    if (update < -clipparameter) { update = -clipparameter; }
                    //Update weight and average
                    Weights[i, ii] -= update;
                    AvgGradient -= update;
                }
                //Normal gradient descent update
                double bgradient = BiasGradient[i] * (-2d / batchsize);
                //Gradient clipping
                if (bgradient > clipparameter) { bgradient = clipparameter; }
                if (bgradient < -clipparameter) { bgradient = -clipparameter; }
                //Definition of RMSProp
                BRMSGrad[i] = (BRMSGrad[i] * RMSDecay) + ((1 - RMSDecay) * (bgradient * bgradient));
                //Update bias
                Biases[i] -= (learningrate / Math.Sqrt(BRMSGrad[i])) * bgradient;
            }
            //Reset gradients (but not RMS gradients)
            WeightGradient = new double[Length, InputLength];
            BiasGradient = new double[Length];
        }
        /// <summary>
        /// Descent for the first layer
        /// </summary>
        /// <param name="input">Original image (optionally normalized)</param>
        public void Descend(double[,] input)
        {
            double[] input2 = new double[input.Length];
            int iterator = 0;
            foreach (double d in input) { input2[iterator] = d; iterator++; }
            Descend(input2, false);
        }
        /// <summary>
        /// Descent for other layers
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="output">Whether the layer is the output layer</param>
        public void Descend(double[] input, bool output)
        {
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Weight gradients
                    WeightGradient[i, ii] = input[ii] * Maths.TanhDerriv(Values[i]) * Errors[i];
                }
                if (output) { continue; }
                //Bias gradients
                BiasGradient[i] = Maths.TanhDerriv(Values[i]) * Errors[i];
            }
        }
        public void Backprop(iLayer output)
        {
            if (output is FullyConnectedLayer)
            {
                var FCLOutput = output as FullyConnectedLayer;
                Errors = new double[Length];
                for (int k = 0; k < FCLOutput.Length; k++)
                {
                    for (int j = 0; j < Length; j++)
                    {
                        Errors[j] += FCLOutput.Weights[k, j] * Maths.TanhDerriv(FCLOutput.Values[k]) * FCLOutput.Errors[k];
                    }
                }
            }
            else
            {
                var CLOutput = output as ConvolutionLayer;
                Errors = new double[Length];
                //Dot product
                for (int x = 0; x < CLOutput.Length; x++)
                {
                    for (int y = 0; y < Length; y++)
                    {
                        //May be done incorrectly (output[y]..?)
                        Errors[y] += CLOutput.Weights[x, y] * Maths.TanhDerriv(CLOutput.ZVals[x]) * CLOutput.Errors[x];
                    }
                }
            }
        }
        public void Backprop(double correct)
        {
            Errors = new double[Length];
            for (int i = 0; i < Length; i++)
            {
                Errors[i] = 2d * (correct - Values[i]);
            }
        }
        public void Calculate(double[] input, bool output)
        {
            Values = new double[Length];
            for (int k = 0; k < Length; k++)
            {
                for (int j = 0; j < InputLength; j++)
                {
                    Values[k] += Weights[k, j] * input[j];
                }
                if (!output)
                {
                    Values[k] += Biases[k];
                }
            }
            if (!output) { Values = Maths.Tanh(Values); }
        }
        public void Calculate(double[,] input)
        {
            Calculate(Maths.Convert(input), false);
        }
    }
}
