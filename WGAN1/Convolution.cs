using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class Convolution
    {
        double[,] Kernel { get; set; }
        double[,] RMSGrad { get; set; }
        double[,] Errors { get; set; }
        double[,] Gradients { get; set; }
        double[,] ZVals { get; set; }
        public double AvgUpdate { get; set; }

        public Convolution(int kernelsizex, int kernelsizey)
        {
            Kernel = new double[kernelsizex, kernelsizey];
            RMSGrad = new double[kernelsizex, kernelsizey];
            Gradients = new double[kernelsizex, kernelsizey];
            ZVals = new double[kernelsizey, kernelsizey];
        }
        public Convolution Init(int ksx, int ksy)
        {
            Kernel = new double[ksx, ksy];
            var r = new Random();
            for (int i = 0; i < ksx; i++)
            {
                for (int ii = 0; ii < ksy; ii++)
                {
                    Kernel[i, ii] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (ksy * ksy));
                }
            }
            return this;
        }
        public void Descend(int batchsize, double learningrate, double clipparameter, double RMSdecay)
        {
            AvgUpdate = 0;
            for (int i = 0; i < Kernel.GetLength(0); i++)
            {
                for (int ii = 0; ii < Kernel.GetLength(1); ii++)
                {
                    double gradient = Gradients[i, ii] * (-2d / batchsize);
                    RMSGrad[i, ii] = (RMSGrad[i, ii] * RMSdecay) + ((1 - RMSdecay) * (gradient * gradient));
                    double update = (learningrate / Math.Sqrt(RMSGrad[i, ii])) * gradient;
                    //Gradient clipping
                    if (update > clipparameter) { update = clipparameter; }
                    if (update < -clipparameter) { update = -clipparameter; }
                    Kernel[i, ii] -= update;
                    AvgUpdate -= update;
                }
            }
            AvgUpdate /= Kernel.Length;
            Gradients = new double[Kernel.GetLength(0), Kernel.GetLength(1)];
        }
        public void Descend(double[] input)
        {
            var kernelsizex = Kernel.GetLength(0);
            var kernelsizey = Kernel.GetLength(1);

            Gradients = new double[kernelsizex, kernelsizey];
            for (int k = 0; k < kernelsizex; k++)
            {
                for (int j = 0; j < kernelsizey; j++)
                {
                    Gradients[k, j] += input[j] * Statistics.TanhDerriv(ZVals[k / 28, k % 28]) * Errors[k / 28, k % 28];
                }
            }
        }
        /// <summary>
        /// Calculates the errors of the convolution
        /// </summary>
        /// <param name="l">The layer which comes after the convolutional layer</param>
        public void Backprop(Layer l)
        {
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
        }
        /// <summary>
        /// Calculates the dot product of the kernel and input matrix.
        /// Matrices should be size [x, y] and [y], respectively, where x is the output size and y is the latent space's size
        /// </summary>
        /// <param name="input">The input matrix</param>
        /// <returns></returns>
        public double[] CalcDotProduct(double[] input)
        {
            int xsize = Kernel.GetLength(0);
            int ysize = input.Length;
            if (Kernel.GetLength(1) != ysize) { throw new Exception("Invalid matrix sizes"); }
            var output = new double[ysize * ysize];
            //Dot product
            for (int x = 0; x < xsize; x++)
            {
                for (int y = 0; y < ysize; y++)
                {
                    //May be done incorrectly (output[y]..?)
                    output[x] += Kernel[x, y] * input[y];
                }
            }
            ZVals = new double[ysize, ysize];
            int iterator = 0;
            for (int i = 0; i < ysize; i++)
            {
                for (int ii = 0; ii < ysize; ii++)
                {
                    ZVals[i, ii] = output[iterator]; iterator++;
                }
            }
            return output;
        }
    }
}
