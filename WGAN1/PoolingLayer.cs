using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class PoolingLayer : iLayer
    {
        public double[,] Weights { get; set; }
        public double[] ZVals { get; set; }
        public double[] Errors { get; set; }
        public int OutputLength { get; set; }
        public int Length { get; set; }
        public int InputLength { get; set; }
        public bool UsesTanh { get; set; }
        public int PoolSize { get; set; }
        public bool DownOrUp { get; set; }
        public PoolingLayer(bool downorup, int l, int il)
        {
            PoolSize = l; InputLength = il;
            DownOrUp = downorup;
            if (downorup) 
            { 
                if ((int)Math.Sqrt(il) != Math.Sqrt(il)) { throw new Exception("Invalid input size (non-sqrtable)"); }
                if (il % l != 0) { throw new Exception("Invalid pool size (unclean divisor)"); }
                Length = (int)Math.Pow(Math.Sqrt(il) / l, 2); 
            }
            else { Length = (int)Math.Pow(Math.Sqrt(il) * l, 2); }
        }
        public void Backprop(double[] input, iLayer outputlayer, double[] outputvals, double correct, bool calcgradients)
        {
            //Calculate errors
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
                }
            }
            if (outputlayer is PoolingLayer)
            {
                var PLOutput = outputlayer as PoolingLayer;
                if (PLOutput.DownOrUp)
                {
                    int iterator = 0;
                    Errors = new double[Length];
                    var wets = Maths.Convert(PLOutput.Weights);
                    for (int i = 0; i < Length; i++)
                    {
                        if (wets[i] == 0) { continue; }
                        Errors[i] = PLOutput.Errors[iterator];
                        iterator++;
                    }
                    if (outputlayer.UsesTanh) { Errors = Maths.TanhDerriv(Errors); }
                    //Otherwise they're fine as-is
                }
                else
                {
                    PLOutput.Calculate(PLOutput.Errors, false);
                    if (outputlayer.UsesTanh) { Errors = Maths.TanhDerriv(PLOutput.ZVals); }
                    else { Errors = PLOutput.ZVals; }
                }
            }
            //There are no gradients with respect to a pooling layer
        }

        public void Calculate(double[] input, bool output)
        {
            Calculate(Maths.Convert(input), output);
        }

        public void Calculate(double[,] input, bool useless)
        {
            //If pooling down
            if (DownOrUp)
            {
                if (input.GetLength(0) % PoolSize != 0 || input.GetLength(1) % PoolSize != 0)
                { throw new Exception("Unclean divide in PoolSizing"); }
                double[,] output = new double[input.GetLength(0) / PoolSize, input.GetLength(1) / PoolSize];
                var mask = new double[input.GetLength(0), input.GetLength(1)];
                int currentx = 0, currenty;
                for (int i = 0; i < input.GetLength(0); i += PoolSize)
                {
                    currenty = 0;
                    for (int ii = 0; ii < input.GetLength(1); ii += PoolSize)
                    {
                        double max = double.MinValue; int maxX = i, maxY = ii;
                        for (int j = 0; j < PoolSize; j++)
                        {
                            for (int jj = 0; jj < PoolSize; jj++)
                            {
                                if (input[i + j, ii + jj] > max)
                                { max = input[i + j, ii + jj]; maxX = i + j; maxY = ii + jj; continue; }
                            }
                        }
                        mask[maxX, maxY] = 1;
                        output[currentx, currenty] = input[maxX, maxY];
                        currenty++;
                    }
                    currentx++;
                }
                ZVals = Maths.Convert(output);
                Weights = mask;
            }
            //If pooling up
            else
            {
                int xsize = input.GetLength(0) * PoolSize;
                int ysize = input.GetLength(1) * PoolSize;
                double[,] output = new double[xsize, ysize];
                for (int i = 0; i < xsize; i++)
                {
                    for (int ii = 0; ii < ysize; ii++)
                    {
                        output[i, ii] = input[i / PoolSize, ii / PoolSize];
                    }
                }
                ZVals = Maths.Convert(output);
            }
        }

        public void Descend(int batchsize, bool batchnorm)
        {
            throw new NotImplementedException();
        }

        public iLayer Init(bool isoutput)
        {
            return this;
        }
    }
}
