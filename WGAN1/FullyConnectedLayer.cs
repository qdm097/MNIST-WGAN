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
        /// Descent for other layers
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="isoutput">Whether the layer is the output layer</param>
        public void Descend(double[] input, bool isoutput)
        {
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Weight gradients
                    WeightGradient[i, ii] += input[ii] * Maths.TanhDerriv(Values[i]) * Errors[i];
                }
                if (isoutput) { continue; }
                //Bias gradients
                BiasGradient[i] += Errors[i];
            }
        }
        /// <summary>
        /// I used the following intuition to work out this method of backpropegation, 
        /// because I could not find an explanation anywhere online:
        /// "Error is how much you're wrong, adjusted for how much your superior cares and how much he's wrong"
        /// I then realized that this applies to convolution as much as it does normally.
        /// So that means the error, with respect to any given input value, is defined the same as normally.
        /// In other words, <i>you can use the same formula as normal, but calculate it with convolution</i>
        /// This is done like so: "Error += output.weight * output.error * tanhderriv(output.zval)"
        /// With respect to the given indices: i, ii, j, jj.
        /// All adjusted for convolution, demonstraighted below.
        /// </summary>
        /// <param name="outputlayer"></param>
        public void Backprop(iLayer outputlayer)
        {            
            if (outputlayer is FullyConnectedLayer)
            {
                var FCLOutput = outputlayer as FullyConnectedLayer;
                Errors = new double[Length];
                for (int k = 0; k < FCLOutput.Length; k++)
                {
                    for (int j = 0; j < Length; j++)
                    {
                        Errors[j] += FCLOutput.Weights[k, j] * Maths.TanhDerriv(FCLOutput.ZVals[k]) * FCLOutput.Errors[k];
                    }
                }
            }
            else
            {
                var ocl = outputlayer as ConvolutionLayer;
                var sidelength = (int)Math.Sqrt(Length);
                int length = (sidelength / ConvolutionLayer.StepSize) - ocl.KernelSize + 1;
                int width = (sidelength / ConvolutionLayer.StepSize) - ocl.KernelSize + 1;
                int ss = ConvolutionLayer.StepSize;

                var oclerrors = Maths.Convert(ocl.Errors);
                var inputvalues = Maths.Convert(Values);

                double[,] errors = new double[sidelength, sidelength];
                for (int i = 0; i < length; i++)
                {
                    for (int ii = 0; ii < width; ii++)
                    {
                        for (int j = 0; j < ocl.KernelSize; j += ss)
                        {
                            for (int jj = 0; jj < ocl.KernelSize; jj += ss)
                            {
                                //Error += weight * error * tanhderriv(zval)
                                errors[(i * ss) + j, (ii * ss) + jj] += ocl.Weights[j, jj] * oclerrors[j, jj] 
                                    * Maths.TanhDerriv(ocl.Weights[j, jj] * inputvalues[(i * ss) + j, (ii * ss) + jj]);
                            }
                        }
                    }
                }
                Errors = Maths.Convert(errors);
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
