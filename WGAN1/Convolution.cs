using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class ConvolutionLayer : iLayer
    {
        //Kernel
        public double[,] Weights { get; set; }
        public int[,] Mask { get; set; }
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
            Length = kernelsize; InputLength = inputsize;
            KernelSize = (int)Math.Sqrt(kernelsize);
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
        public void Descend(int batchsize, double learningrate, double clipparameter, double RMSdecay)
        {
            AvgUpdate = 0;
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    double gradient = Gradients[i, ii] * (-2d / batchsize);
                    RMSGrad[i, ii] = (RMSGrad[i, ii] * RMSdecay) + ((1 - RMSdecay) * (gradient * gradient));
                    double update = (learningrate / Math.Sqrt(RMSGrad[i, ii])) * gradient;
                    //Gradient clipping
                    if (update > clipparameter) { update = clipparameter; }
                    if (update < -clipparameter) { update = -clipparameter; }
                    Weights[i, ii] -= update;
                    AvgUpdate -= update;
                }
            }
            AvgUpdate /= Weights.Length;
            Gradients = new double[KernelSize, KernelSize];
        }
        public void Descend(double[] input, bool useless)
        {
            Gradients = new double[KernelSize, KernelSize];
            for (int k = 0; k < KernelSize; k++)
            {
                for (int j = 0; j < KernelSize; j++)
                {
                    Gradients[k, j] += input[j] * Maths.TanhDerriv(ZVals[k]) * Errors[k];
                }
            }
        }
        /// <summary>
        /// Calculates the errors of the convolution
        /// </summary>
        /// <param name="outputlayer">The layer which comes after the convolutional layer</param>
        public void Backprop(iLayer outputlayer)
        {
            if (outputlayer is FullyConnectedLayer)
            {
                var errors = new double[outputlayer.InputLength];
                for (int k = 0; k < outputlayer.Length; k++)
                {
                    for (int j = 0; j < outputlayer.InputLength; j++)
                    {
                        errors[j] += outputlayer.Weights[k, j] * Maths.TanhDerriv(outputlayer.ZVals[k]) * outputlayer.Errors[k];
                    }
                }
                //The error of a convolutional matrix is equivalent to 
                //the full convolution of the kernel and its flipped error matrix
                Errors = Maths.Convert(FullConvolve(Maths.Convert(errors)));
            }
            else
            {
                var ocl = outputlayer as ConvolutionLayer;
                var sidelength = (int)Math.Sqrt(Values.Length);
                int length = (sidelength / StepSize) - ocl.KernelSize + 1;
                int width = (sidelength / StepSize) - ocl.KernelSize + 1;
                int ss = StepSize;

                var oclerrors = Maths.Convert(ocl.Errors);
                var inputvalues = Maths.Convert(Values);

                double[,] errors = new double[sidelength, sidelength];
                for (int i = 0; i < length; i++)
                {
                    for (int ii = 0; ii < width; ii++)
                    {
                        for (int j = 0; j < ocl.KernelSize; j++)
                        {
                            for (int jj = 0; jj < ocl.KernelSize; jj++)
                            {
                                //Error += weight * error * tanhderriv(zval)
                                errors[(i * ss) + j, (ii * ss) + jj] += ocl.Weights[j, jj] * oclerrors[j, jj]
                                    * Maths.TanhDerriv(ocl.Weights[j, jj] * inputvalues[(i * ss) + j, (ii * ss) + jj]);
                            }
                        }
                    }
                }
                Errors = Maths.Convert(FullConvolve(errors));
            }
        }
        public void Backprop(double useless) { throw new Exception("The convolution layer is never an output layer"); }
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
        /// <summary>
        /// Partial convolution layer
        /// </summary>
        /// <param name="input"></param>
        /// <param name="isoutput"></param>
        public void Calculate(double[,] input, bool isoutput)
        {
            //Critic layer doesn't use masks or padding
            var output = COG ? Convolve(input) : Convolve(Pad(input));
            ZVals = Maths.Convert(output);
            if (!isoutput) { output = Maths.Tanh(output); }
            Values = Maths.Convert(output);
        }
        public double[,] Convolve(double[,] input)
        {
            int length = (input.GetLength(0) / StepSize) - KernelSize + 1;
            int width = (input.GetLength(1) / StepSize) - KernelSize + 1;

            double[,] output = new double[length, width];
            for (int i = 0; i < length; i++)
            {
                for (int ii = 0; ii < width; ii++)
                {
                    for (int j = 0; j < KernelSize; j += StepSize)
                    {
                        for (int jj = 0; jj < KernelSize; jj += StepSize)
                        {
                            //Only add the value if it's an original value (Mask[_,_] != 0)
                            if (!COG && Mask[(i * StepSize) + j, (ii * StepSize) + jj] == 0) { continue; }
                            output[i, ii] += input[(i * StepSize) + j, (ii * StepSize) + jj] * Weights[j, jj];
                        }
                    }
                }
            }
            return output;
        }
        public double[,] FullConvolve(double[,] input)
        {
            double[,] output = new double[KernelSize, KernelSize];
            for (int i = 0; i < KernelSize; i++)
            {
                for (int ii = 0; ii < KernelSize; ii++)
                {
                    for (int j = 0; j < input.GetLength(0); j++)
                    {
                        for (int jj = 0; jj < input.GetLength(1); jj++)
                        {
                            if (i + j >= KernelSize || ii + jj >= KernelSize) { continue; }
                            output[i, ii] += Weights[(i * StepSize) + j, (ii * StepSize) + jj] * input[j, jj];
                        }
                    }
                }
            }
            return output;
        }
        public double[,] FullConvolveTest(double[,] input)
        {
            var size = 6;
            double[,] output = new double[size, size];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int ii = 0; ii < input.GetLength(1); ii++)
                {
                    for (int j = 0; j < KernelSize; j++)
                    {
                        for (int jj = 0; jj < KernelSize; jj++)
                        {
                            if (i + j >= size || ii + jj >= size) { continue; }
                            output[i, ii] += input[(i * StepSize) + j, (ii * StepSize) + jj] * Weights[j, jj];
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
            Mask = new int[inputxsize + (2 * padsize), inputysize + (2 * padsize)];

            for (int i = 0; i < inputxsize; i++)
            {
                for (int ii = 0; ii < inputysize; ii++)
                {
                    output[i + padsize, ii + padsize] = input[i, ii]; Mask[i + padsize, ii + padsize] = 1;
                }
            }
            return output;
        }
    }
}
