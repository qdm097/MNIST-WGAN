using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Runtime.Remoting.Metadata;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    public abstract class Layer
    {
        public double[,] Weights { get; set; }
        public List<double[]> Errors { get; set; }
        public List<double[]> Values { get; set; }
        public List<double[]> ZVals { get; set; }
        public int OutputLength { get; set; }
        public int Length { get; set; }
        public int InputLength { get; set; }
        public int ActivationFunction { get; set; }
        public abstract Layer Init(bool isoutput);
        public abstract void Descend(bool batchnorm);
        /// <summary>
        /// Descent for other layers
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="output">Whether the layer is the output layer</param>
        public void Backprop(List<double[]> inputs, Layer outputlayer, double loss, bool calcgradients)
        {
            Errors = new List<double[]>();

            //Calculate errors
            if (outputlayer is null)
            {
                for (int j = 0; j < inputs.Count; j++)
                {
                    Errors.Add(new double[Length]);
                    for (int i = 0; i < Length; i++)
                    {
                        //(i == loss ? 1d : 0d)
                        Errors[j][i] = 2d * (Values[j][i] - loss);
                    }
                }
            }
            else
            {
                for (int i = 0; i < inputs.Count; i++)
                {
                    Errors.Add(new double[outputlayer.InputLength]);
                }
                if (outputlayer is SumLayer)
                {
                    //Errors with respect to the output of the convolution
                    //dl/do
                    for (int i = 0; i < outputlayer.ZVals.Count; i++)
                    {
                        for (int k = 0; k < outputlayer.Length; k++)
                        {
                            for (int j = 0; j < outputlayer.InputLength; j++)
                            {
                                Errors[i][j] += outputlayer.Errors[i][k];
                            }
                        }
                    }
                }

                //Apply tanhderriv, if applicable, to the output's zvals
                var outputZVals = outputlayer.ZVals;
                if (outputlayer.ActivationFunction == 0) { outputZVals = Maths.TanhDerriv(outputlayer.ZVals); }
                if (outputlayer.ActivationFunction == 1) { outputZVals = Maths.ReLuDerriv(outputlayer.ZVals); }

                if (outputlayer is FullyConnectedLayer)
                {
                    var FCLOutput = outputlayer as FullyConnectedLayer;
                    for (int i = 0; i < outputlayer.ZVals.Count; i++)
                    {
                        for (int k = 0; k < FCLOutput.Length; k++)
                        {
                            for (int j = 0; j < FCLOutput.InputLength; j++)
                            {
                                Errors[i][j] += FCLOutput.Weights[k, j] * outputZVals[i][k] * FCLOutput.Errors[i][k];
                            }
                        }
                    }
                }
                if (outputlayer is ConvolutionLayer)
                {
                    var CLOutput = outputlayer as ConvolutionLayer;
                    for (int i = 0; i < outputlayer.ZVals.Count; i++)
                    {
                        if ((outputlayer as ConvolutionLayer).DownOrUp)
                        {
                            Errors[i] = Maths.Convert(CLOutput.UnPad(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors[i]))));
                        }
                        else
                        {
                            Errors[i] = Maths.Convert(CLOutput.UnPad(CLOutput.Convolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors[i]))));
                        }
                    }

                    //Errors = Maths.Convert(CLOutput.UnPad(CLOutput.FullConvolve(CLOutput.Weights, Maths.Convert(CLOutput.Errors))));
                }
                if (outputlayer is PoolingLayer)
                {
                    var PLOutput = outputlayer as PoolingLayer;
                    for (int b = 0; b < NN.BatchSize; b++)
                    {
                        if (PLOutput.DownOrUp)
                        {
                            int iterator = 0;
                            var wets = Maths.Convert(PLOutput.Weights);
                            for (int i = 0; i < Length; i++)
                            {
                                if (wets[i] == 0) { continue; }
                                Errors[b][i] = PLOutput.Errors[b][iterator];
                                iterator++;
                            }
                        }
                        else
                        {
                            //Sum the errors
                            double[,] outputerrors = Maths.Convert(PLOutput.Errors[b]);
                            int oel = outputerrors.GetLength(0);
                            int oew = outputerrors.GetLength(1);
                            double[,] errors = new double[oel / PLOutput.PoolSize, oew / PLOutput.PoolSize];
                            for (int i = 0; i < oel; i++)
                            {
                                for (int ii = 0; ii < oew; ii++)
                                {
                                    errors[i / PLOutput.PoolSize, ii / PLOutput.PoolSize] += outputerrors[i, ii];
                                }
                            }
                            Errors[b] = Maths.Convert(errors);
                        }
                    }
                }
            }
            if (NN.NormErrors && Errors[0].Length > 1)
            {
                Errors = Maths.Normalize(Errors);
            }
            if (calcgradients)
            {
                if (this is FullyConnectedLayer) { (this as FullyConnectedLayer).CalcGradients(inputs, outputlayer); }
                if (this is ConvolutionLayer) { (this as ConvolutionLayer).CalcGradients(inputs, outputlayer); }
                if (this is PoolingLayer) { return; }
                if (this is SumLayer) { return; }
            }
        }
        public abstract void CalcGradients(List<double[]> inputs, Layer outputlayer);
        public abstract void Calculate(List<double[]> inputs, bool output);
    }
}
