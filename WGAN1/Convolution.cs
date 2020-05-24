using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class ConvolutionLayer : iLayer
    {
        //Kernel
        public double[,] Weights { get; set; }
        //Whether this layer belongs to a [C]ritic [O]r [G]enerator
        public bool COG { get; set; }
        public int Length { get; set; }
        public int KernelSize { get; set; }
        public int InputLength { get; set; }
        double[,] RMSGrad { get; set; }
        public double[] Errors { get; set; }
        double[,] Gradients { get; set; }
        public double[] ZVals { get; set; }
        public double[] Values { get; set; }
        public double AvgUpdate { get; set; }
        public static int StepSize = 1;

        public ConvolutionLayer(int kernelsize, int inputsize)
        {
            InputLength = inputsize;
            Length = kernelsize * kernelsize;
            KernelSize = kernelsize;
            Weights = new double[KernelSize, KernelSize];
            RMSGrad = new double[KernelSize, KernelSize];
            Gradients = new double[KernelSize, KernelSize];
        }
        public iLayer Init(bool useless)
        {
            Weights = new double[KernelSize, KernelSize];
            var r = new Random();
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    Weights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (InputLength * InputLength));
                }
            }
            return this;
        }
        public void Descend(int batchsize)
        {
            AvgUpdate = 0;
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    double gradient = Gradients[i, ii] * (-2d / batchsize); 
                    double update = NN.LearningRate * gradient;
                    //Root mean square propegation
                    if (NN.UseRMSProp)
                    {
                        RMSGrad[i, ii] = (RMSGrad[i, ii] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (gradient * gradient));
                        update = (NN.LearningRate / Math.Sqrt(RMSGrad[i, ii])) * gradient;
                    }
                    //Gradient clipping
                    if (NN.UseClipping)
                    {
                        if (update > NN.ClipParameter) { update = NN.ClipParameter; }
                        if (update < -NN.ClipParameter) { update = -NN.ClipParameter; }
                    }
                    Weights[i, ii] -= update;
                    AvgUpdate -= update;
                }
            }
            Gradients = new double[KernelSize, KernelSize];
        }
        public void Backprop(double[] input, iLayer outputlayer, bool isoutput, double correct, bool calcgradients)
        {
            //Calc errors
            double[,] Input = Maths.Convert(input);
            if (outputlayer is FullyConnectedLayer)
            {
                //Errors with respect to the output of the convolution
                //dl/do
                Errors = new double[outputlayer.InputLength];
                for (int k = 0; k < outputlayer.Length; k++)
                {
                    for (int j = 0; j < outputlayer.InputLength; j++)
                    {
                        Errors[j] += outputlayer.Weights[k, j] * Maths.TanhDerriv(outputlayer.ZVals[k]) * outputlayer.Errors[k];
                    }
                }
            }
            if (outputlayer is ConvolutionLayer)
            {
                
                var CLOutput = outputlayer as ConvolutionLayer;
                //Errors = Maths.Convert(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)));
                
                //Critic upscales to find errors
                if ((outputlayer as ConvolutionLayer).COG) { Errors = Maths.Convert(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors))); }
                //Generator downscales to find them
                else { Errors = Maths.Convert(Flip(CLOutput.Convolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)))); }
                
            }
            //Gradients = Convolve(Maths.Convert(Errors), Input);

            if (COG && calcgradients) { Gradients = Convolve(Maths.Convert(Errors), Input); }
            //No idea if this is accurate, but it works
            if (!COG && calcgradients)
            {
                Gradients = Convolve(Input, Maths.Convert(Errors));
            }

        }
        /// <summary>
        /// Calculates the dot product of the kernel and input matrix.
        /// Matrices should be size [x, y] and [y], respectively, where x is the output size and y is the latent space's size
        /// </summary>
        /// <param name="input">The input matrix</param>
        /// <param name="isoutput">Whether to use hyperbolic tangent on the output</param>
        /// <returns></returns>
        public void Calculate(double[] input, bool isoutput)
        {
            Calculate(Maths.Convert(input), isoutput);
        }
        public void Calculate(double[,] input, bool isoutput)
        {
            //Padded upscaling
            //var output = COG ? Convolve(Weights, input) : Convolve(Weights, Pad(input));

            //Transposed convolutional upscaling
            var output = COG ? Convolve(Weights, input) : FullConvolve(Weights, input);
            ZVals = Maths.Convert(output);
            if (!isoutput) { output = Maths.Tanh(output); }
            Values = Maths.Convert(output);
        }
        public double[,] Convolve(double[,] filter, double[,] input)
        {
            int kernelsize = filter.GetLength(0);
            int length = (input.GetLength(0) / StepSize) - kernelsize + 1;
            int width = (input.GetLength(1) / StepSize) - kernelsize + 1;
            double[,] output = new double[length, width];
            for (int i = 0; i < length; i++)
            {
                for (int ii = 0; ii < width; ii++)
                {
                    for (int j = 0; j < kernelsize; j += StepSize)
                    {
                        for (int jj = 0; jj < kernelsize; jj += StepSize)
                        {
                            output[i, ii] += input[(i * StepSize) + j, (ii * StepSize) + jj] * filter[j, jj];
                        }
                    }
                }
            }
            return output;
        }
        /// <summary>
        /// This is also known as "Transposed convolution" and "Partially strided convolution"
        /// </summary>
        /// <param name="filter"></param>
        /// <param name="input"></param>
        /// <returns></returns>
        public double[,] FullConvolve(double[,] filter, double[,] input)
        {
            var kernelsize = input.GetLength(0) + filter.GetLength(0) - 1;
            double[,] output = new double[kernelsize, kernelsize];
            for (int i = 0; i < input.GetLength(0); i += StepSize)
            {
                for (int ii = 0; ii < input.GetLength(1); ii += StepSize)
                {
                    for (int j = 0; j < filter.GetLength(0); j += StepSize)
                    {
                        for (int jj = 0; jj < filter.GetLength(1); jj += StepSize)
                        {
                            if ((i * StepSize) + j >= kernelsize || (ii * StepSize) + jj >= kernelsize) { continue; }
                            output[(i * StepSize) + j, (ii * StepSize) + jj] += input[i, ii] * filter[j, jj];
                        }
                    }
                }
            }
            return output;
        }
        public double[,] Flip(double[,] input)
        {
            int length = input.GetLength(0);
            int width = input.GetLength(1);
            var output = new double[length, width];
            for (int i = 0; i < length; i++)
            {
                for (int ii = 0; ii < width; ii++)
                {
                    output[ii, i] = input[i, ii];
                }
            }
            return output;
        }
        public double[,] Pad(double[,] input)
        {
            int inputxsize = input.GetLength(0);
            int inputysize = input.GetLength(1);
            int padsize = KernelSize - 1;

            var output = new double[inputxsize + (2 * padsize), inputysize + (2 * padsize)];

            for (int i = 0; i < inputxsize; i++)
            {
                for (int ii = 0; ii < inputysize; ii++)
                {
                    output[i + padsize, ii + padsize] = input[i, ii];
                }
            }
            return output;
        }
    }
}
