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
        double[,] RMSGrad { get; set; }
        double[,] Gradients { get; set; }
        double[,] Updates { get; set; }
        //Whether this layer belongs to a [C]ritic [O]r [G]enerator
        public bool DownOrUp { get; set; }
        public int Length { get; set; }
        public int KernelSize { get; set; }
        public int InputLength { get; set; }
        public double[] Errors { get; set; }
        public double[] ZVals { get; set; }
        public double[] Values { get; set; }
        public double AvgUpdate { get; set; }
        public int Stride { get; set; }

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
        public void Descend(int batchsize, bool batchnorm)
        {
            //Calculate gradients
            Updates = new double[KernelSize, KernelSize];
            AvgUpdate = 0;
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    Updates[i, ii] = NN.LearningRate * Gradients[i, ii] * (-2d / batchsize);
                    //Root mean square propegation
                    if (NN.UseRMSProp)
                    {
                        RMSGrad[i, ii] = (RMSGrad[i, ii] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (Updates[i, ii] * Updates[i, ii]));
                        Updates[i, ii] = (NN.LearningRate / Math.Sqrt(RMSGrad[i, ii])) * Updates[i, ii];
                    }
                }
            }
            //Gradient normalization
            if (batchnorm) { Updates = Maths.Scale(NN.LearningRate, Maths.Normalize(Updates)); }
            //Apply updates
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    Weights[i, ii] -= Updates[i, ii];
                    AvgUpdate -= Updates[i, ii];
                    //Gradient clipping
                    if (NN.UseClipping)
                    {
                        if (Weights[i, ii] > NN.ClipParameter) { Weights[i, ii] = NN.ClipParameter; }
                        if (Weights[i, ii] < -NN.ClipParameter) { Weights[i, ii] = -NN.ClipParameter; }
                    }
                }
            }
            Gradients = new double[KernelSize, KernelSize];
        }
        public void Backprop(double[] input, iLayer outputlayer, bool isoutput, double correct, bool calcgradients)
        {
            //Calculate error
            if (isoutput && outputlayer is null)
            {
                //Leveraging the fact that only the critic uses this formula,
                //and the critic always has an output size of [1]
                Errors = new double[1] { 2d * (correct - Values[0])};
            }
            else
            {
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
                    if ((outputlayer as ConvolutionLayer).DownOrUp) { Errors = Maths.Convert(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors))); }
                    //Generator downscales to find them
                    else { Errors = Maths.Convert(Flip(CLOutput.Convolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)))); }

                }
                //Gradients = Convolve(Maths.Convert(Errors), Input);
            }

            if (calcgradients) 
            { 
                double[,] Input = Maths.Convert(input);
                if (DownOrUp) { Gradients = Convolve(Maths.Convert(Errors), Input); }
                //No idea if this is accurate, but it works
                else { Gradients = Convolve(Input, Maths.Convert(Errors)); }
            }
        }
        /// <summary>
        /// Calculates the dot product of the kernel and input matrix.
        /// Matrices should be size [x, y] and [y], respectively, where x is the output size and y is the latent space's size
        /// </summary>
        /// <param name="input">The input matrix</param>
        /// <param name="isoutput">Whether to use hyperbolic tangent on the output</param>
        /// <returns></returns>
        public void Calculate(double[] input, bool isoutput, bool usetanh)
        {
            Calculate(Maths.Convert(input), isoutput, usetanh);
        }
        public void Calculate(double[,] input, bool isoutput, bool usetanh)
        {
            //Padded upscaling
            //var output = COG ? Convolve(Weights, input) : Convolve(Weights, Pad(input));

            //Transposed convolutional upscaling
            var output = DownOrUp ? Convolve(Weights, input) : FullConvolve(Weights, input);
            ZVals = Maths.Convert(output);
            if (!isoutput && usetanh) { output = Maths.Tanh(output); }
            Values = Maths.Convert(output);
        }
        public double[,] Convolve(double[,] filter, double[,] input)
        {
            int kernelsize = filter.GetLength(0);
            int length = ((input.GetLength(0) - kernelsize) / Stride) + 1;
            int width = ((input.GetLength(1) - kernelsize) / Stride) + 1;
            double[,] output = new double[length, width];
            for (int i = 0; i < length; i += Stride)
            {
                for (int ii = 0; ii < width; ii += Stride)
                {
                    for (int j = 0; j < kernelsize; j++)
                    {
                        for (int jj = 0; jj < kernelsize; jj++)
                        {
                            output[i, ii] += input[(i * Stride) + j, (ii * Stride) + jj] * filter[j, jj];
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
            var kernelsize = (Stride * (input.GetLength(0) - 1)) + filter.GetLength(0);
            double[,] output = new double[kernelsize, kernelsize];
            for (int i = 0; i < input.GetLength(0); i += Stride)
            {
                for (int ii = 0; ii < input.GetLength(1); ii += Stride)
                {
                    for (int j = 0; j < filter.GetLength(0); j++)
                    {
                        for (int jj = 0; jj < filter.GetLength(1); jj++)
                        {
                            if ((i * Stride) + j >= kernelsize || (ii * Stride) + jj >= kernelsize) { continue; }
                            output[(i * Stride) + j, (ii * Stride) + jj] += input[i, ii] * filter[j, jj];
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
