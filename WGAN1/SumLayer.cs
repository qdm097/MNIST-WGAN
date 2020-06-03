using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class SumLayer : iLayer
    {
        public double[,] Weights { get { throw new NotImplementedException(); } set { throw new NotImplementedException(); } }
        public double[] ZVals { get; set; }
        public double[] Errors { get; set; }
        public int OutputLength { get; set; }
        public int Length { get; set; }
        public int InputLength { get; set; }
        public bool UsesTanh { get; set; }

        public SumLayer(int l, int il)
        {
            Length = l; 
            InputLength = il;
        }
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
                    //Errors with respect to the output of the convolution
                    //dl/do
                    for (int k = 0; k < outputlayer.Length; k++)
                    {
                        for (int j = 0; j < outputlayer.InputLength; j++)
                        {
                            double zvalderriv = outputlayer.ZVals[k];
                            if (outputlayer.UsesTanh) { zvalderriv = Maths.TanhDerriv(outputlayer.ZVals[k]); }
                            Errors[j] += outputlayer.Weights[k, j] * zvalderriv * outputlayer.Errors[k];
                        }
                    }
                }
                if (outputlayer is ConvolutionLayer)
                {
                    var CLOutput = outputlayer as ConvolutionLayer;
                    if ((outputlayer as ConvolutionLayer).DownOrUp) { Errors = Maths.Convert(CLOutput.UnPad(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)))); }
                    else { Errors = Maths.Convert(CLOutput.UnPad(CLOutput.Convolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors)))); }
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
                    }
                    else
                    {
                        PLOutput.Calculate(PLOutput.Errors, false);
                        Errors = PLOutput.ZVals;
                    }
                }
            }
        }

        public void Calculate(double[] input, bool output)
        {
            throw new NotImplementedException();
        }

        public void Calculate(double[,] input, bool output)
        {
            throw new NotImplementedException();
        }
        public void Calculate(double[] input1, double[] input2)
        {
            if (input1.GetLength(0) != input2.GetLength(0))
            {
                throw new Exception("Array sizes do not match");
            }

            double[] output = new double[input1.Length];
            for (int i = 0; i < input1.GetLength(0); i++)
            {
                output[i] = input1[i] + input2[i];
            }
            ZVals = output;
        }
        public void Descend(int batchsize, bool batchnorm)
        {
            throw new NotImplementedException();
        }

        public iLayer Init(bool b) { return this; }
    }
}
