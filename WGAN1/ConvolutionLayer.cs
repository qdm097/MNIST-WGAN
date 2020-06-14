using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class ConvolutionLayer : Layer
    {
        double[,] RMSGrad { get; set; }
        double[,] Gradients { get; set; }
        double[,] Updates { get; set; }
        public int KernelSize { get; set; }
        public double AvgUpdate { get; set; }
        public int Stride { get; set; }
        public int PadSize { get; set; }
        public bool DownOrUp { get; set; }

        public ConvolutionLayer(int kernelsize, int inputsize)
        {
            InputLength = inputsize;
            Length = kernelsize * kernelsize;
            KernelSize = kernelsize;
            Weights = new double[KernelSize, KernelSize];
            RMSGrad = new double[KernelSize, KernelSize];
            Gradients = new double[KernelSize, KernelSize];
        }
        public override Layer Init(bool useless)
        {
            Weights = new double[KernelSize, KernelSize];
            var r = new Random();
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    //LeCun initialization
                    Weights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (InputLength * InputLength));
                }
            }
            return this;
        }
        public override void Descend()
        {
            //Calculate gradients
            Updates = new double[KernelSize, KernelSize];
            AvgUpdate = 0;
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    Updates[i, ii] = Gradients[i, ii] * (2d / NN.BatchSize);
                    //Root mean square propegation
                    if (NN.UseRMSProp)
                    {
                        RMSGrad[i, ii] = (RMSGrad[i, ii] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (Updates[i, ii] * Updates[i, ii]));
                        Updates[i, ii] = (Updates[i, ii] / (Math.Sqrt(RMSGrad[i, ii])/* + NN.Infinitesimal*/));
                    }
                    Updates[i, ii] *= NN.LearningRate;
                }
            }
            //Gradient normalization
            if (NN.NormGradients) { Updates = Maths.Scale(NN.LearningRate, Maths.Normalize(Updates)); }
            //Apply updates
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    Weights[i, ii] -= Updates[i, ii];
                    AvgUpdate -= Updates[i, ii];
                    //Weight clipping
                    if (NN.UseClipping)
                    {
                        if (Weights[i, ii] > NN.ClipParameter) { Weights[i, ii] = NN.ClipParameter; }
                        if (Weights[i, ii] < -NN.ClipParameter) { Weights[i, ii] = -NN.ClipParameter; }
                    }
                }
            }
            Gradients = new double[KernelSize, KernelSize];
        }
        public override void CalcGradients(List<double[]> inputs, Layer outputlayer)
        {
            for (int b = 0; b < NN.BatchSize; b++)
            {
                //var input = inputs[i];
                //if (UsesTanh) { input = Maths.TanhDerriv(inputs[i]); }

                double[,] Input = Pad(Maths.Convert(inputs[b]));
                double[,] stochgradients;
                if (DownOrUp) { stochgradients = Convolve(Maths.Convert(Errors[b]), Input); }
                else { stochgradients = Convolve(Input, Maths.Convert(Errors[b])); }
                //Gradients = stochgradients;

                //Add the stochastic gradients to the batch gradients
                for (int j = 0; j < Gradients.GetLength(0); j++)
                {
                    for (int k = 0; k < Gradients.GetLength(1); k++)
                    {
                        Gradients[j, k] += stochgradients[j, k];
                    }
                }
            }
        }
        /// <summary>
        /// Calculates the dot product of the kernel and input matrix.
        /// Matrices should be size [x, y] and [y], respectively, where x is the output size and y is the latent space's size
        /// </summary>
        /// <param name="inputs">The input matrix</param>
        /// <param name="isoutput">Whether to use hyperbolic tangent on the output</param>
        /// <returns></returns>
        public override void Calculate(List<double[]> inputs, bool isoutput)
        {
            ZVals = new List<double[]>();
            for (int b = 0; b < NN.BatchSize; b++)
            {
                ZVals.Add(Maths.Convert(DownOrUp ? Convolve(Weights, Pad(Maths.Convert(inputs[b]))) : FullConvolve(Weights, Pad(Maths.Convert(inputs[b])))));
            }
            //If normalizing, do so, but only if it won't return an all-zero matrix
            if (NN.NormOutputs && ZVals[0].Length > 1) { ZVals = Maths.Normalize(ZVals); }
            //Use the specified type of activation function
            if (ActivationFunction == 0) { Values = Maths.Tanh(ZVals); return; }
            if (ActivationFunction == 1) { Values = Maths.ReLu(ZVals); return; }
            Values = ZVals; 
        }
        /// <summary>
        /// Convolves a filter across a matrix
        /// </summary>
        /// <param name="filter">Filter matrix</param>
        /// <param name="input">Input matrix</param>
        /// <returns></returns>
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
        /// This is also known as "transposed convolution," "partially strided convolution" and "fractionally strided convolution."
        /// The last term explains it most clearly in my current eyes.
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
        /// <summary>
        /// Partial convolution based padding
        /// </summary>
        /// <param name="input">The matrix to be padded</param>
        /// <returns></returns>
        double[,] Pad(double[,] input)
        {
            if (PadSize == 0) { return input; }

            int inputxsize = input.GetLength(0);
            int inputysize = input.GetLength(1);
            int padxsize = 2 * PadSize;
            int padysize = 2 * PadSize;

            //Create and initialize all-1 matrix
            double[,] filter = new double[padxsize, padysize];
            for (int i = 0; i < padxsize; i++)
            {
                for (int ii = 0; ii < padysize; ii++)
                {
                    filter[i, ii] = 1;
                }
            }

            //Partially convolve all-1 matrix with input matrix
            var output = new double[inputxsize + (2 * PadSize), inputysize + (2 * PadSize)];
            for (int i = 0; i < input.GetLength(0); i += Stride)
            {
                for (int ii = 0; ii < input.GetLength(1); ii += Stride)
                {
                    for (int j = 0; j < filter.GetLength(0); j++)
                    {
                        for (int jj = 0; jj < filter.GetLength(1); jj++)
                        {
                            if (i + j >= inputxsize + padxsize || ii + jj >= inputysize + padysize) { continue; }
                            output[(i * Stride) + j, (ii * Stride) + jj] += input[i, ii] * filter[j, jj];
                        }
                    }
                }
            }

            return output;
        }
        /// <summary>
        /// Convolution based unpadding
        /// </summary>
        /// <param name="input">The matrix to be unpadded</param>
        /// <returns></returns>
        public double[,] UnPad(double[,] input)
        {
            if (PadSize == 0) { return input; }

            int outputxsize = input.GetLength(0) - (2 * PadSize);
            int outputysize = input.GetLength(1) - (2 * PadSize);

            //Create and initialize all-1 matrix
            double[,] filter = new double[2 * PadSize, 2 * PadSize];
            for (int i = 0; i < filter.GetLength(0); i++)
            {
                for (int ii = 0; ii < filter.GetLength(1); ii++)
                {
                    filter[i, ii] = 1;
                }
            }

            //Convolve all-1 matrix with input matrix
            double[,] output = new double[outputxsize, outputysize];
            for (int i = 0; i < outputxsize; i += Stride)
            {
                for (int ii = 0; ii < outputysize; ii += Stride)
                {
                    for (int j = 0; j < filter.GetLength(0); j++)
                    {
                        for (int jj = 0; jj < filter.GetLength(1); jj++)
                        {
                            output[i, ii] += input[(i * Stride) + j, (ii * Stride) + jj] * filter[j, jj];
                        }
                    }
                }
            }

            return output;
        }
    }
}
