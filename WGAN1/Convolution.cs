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
        public int Length { get; set; }
        public int InputLength { get; set; }
        double[,] RMSGrad { get; set; }
        public double[,] Errors { get; set; }
        double[,] Gradients { get; set; }
        public double[,] ZVals { get; set; }
        public double[] Values { get; set; }
        public double AvgUpdate { get; set; }

        public ConvolutionLayer(int kernelsizex, int kernelsizey)
        {
            Length = kernelsizex; InputLength = kernelsizey;
            Weights = new double[kernelsizex, kernelsizey];
            RMSGrad = new double[kernelsizex, kernelsizey];
            Gradients = new double[kernelsizex, kernelsizey];
            ZVals = new double[kernelsizey, kernelsizey];
        }
        public iLayer Init(bool useless)
        {
            Weights = new double[Length, InputLength];
            var r = new Random();
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    Weights[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (Length * InputLength));
                }
            }
            return this;
        }
        public void Descend(int batchsize, double learningrate, double clipparameter, double RMSdecay)
        {
            AvgUpdate = 0;
            for (int i = 0; i < Weights.GetLength(0); i++)
            {
                for (int ii = 0; ii < Weights.GetLength(1); ii++)
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
            Gradients = new double[Weights.GetLength(0), Weights.GetLength(1)];
        }
        public void Descend(double[] input, bool useless)
        {

            Gradients = new double[Length, InputLength];
            for (int k = 0; k < Length; k++)
            {
                for (int j = 0; j < InputLength; j++)
                {
                    Gradients[k, j] += input[j] * Maths.TanhDerriv(ZVals[k / 28, k % 28]) * Errors[k / 28, k % 28];
                }
            }
        }
        public void Descend(double[,] input)
        {
            Descend(Maths.Convert(input), false);
        }
        /// <summary>
        /// Calculates the errors of the convolution
        /// </summary>
        /// <param name="l">The layer which comes after the convolutional layer</param>
        public void Backprop(iLayer l)
        {
            if (l is FullyConnectedLayer)
            {
                var fcl = l as FullyConnectedLayer;

                //Dot product
                var temperrors = new double[Length];
                for (int x = 0; x < fcl.Length; x++)
                {
                    for (int y = 0; y < Length; y++)
                    {
                        temperrors[y] += fcl.Weights[x, y] * Maths.TanhDerriv(fcl.Values[x]) * fcl.Errors[x];
                    }
                }
                //Convert to 2d array
                Errors = Maths.Convert(temperrors);
            }
            else
            {
                throw new Exception("Double conv layers not supported yet");
            }
           
            /*
            //Calc 1d errors
            double[] temp = new double[l.InputLength];
            for (int k = 0; k < l.Length; k++)
            {
                for (int j = 0; j < l.InputLength; j++)
                {
                    temp[j] += l.Weights[k, j] * Statistics.TanhDerriv(l.Values[k]) * l.Errors[k];
                }
            }
            //Convert to 2d array
            double[,] convertederrors = new double[Kernel.GetLength(1), Kernel.GetLength(1)];
            int iterator = 0;
            for (int i = 0; i < Kernel.GetLength(1); i++)
            {
                for (int ii = 0; ii < Kernel.GetLength(1); ii++)
                {
                    convertederrors[i, ii] = temp[iterator]; iterator++;
                }
            }
            Errors = convertederrors;
            */
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
            if (Weights.GetLength(1) != InputLength) { throw new Exception("Invalid matrix sizes"); }
            Values = new double[InputLength * InputLength];
            //Dot product
            for (int x = 0; x < Length; x++)
            {
                for (int y = 0; y < InputLength; y++)
                {
                    //May be done incorrectly (output[y]..?)
                    Values[x] += Weights[x, y] * input[y];
                }
            }
            ZVals = Maths.Convert(Values);
            if (!isoutput) { Values = Maths.Tanh(Values); }
        }
        public void Calculate(double[,] input)
        {
            Calculate(Maths.Convert(input), false);
        }
    }
}
