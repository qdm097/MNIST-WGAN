using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class PoolingLayer : Layer
    {
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

        public override void Calculate(List<double[]> inputs, bool output)
        {
            ZVals = new List<double[]>();
            for (int i = 0; i < NN.BatchSize; i++)
            {
                ZVals.Add(Maths.Convert(Pool(Maths.Convert(inputs[i]), output)));
            }
            if (NN.NormOutputs && ZVals[0].Length > 1) { ZVals = Maths.Normalize(ZVals); }
            if (ActivationFunction == 0) { Values = Maths.Tanh(ZVals); return; }
            if (ActivationFunction == 1) { Values = Maths.ReLu(ZVals); return; }
            Values = ZVals; 
        }

        public double[,] Pool(double[,] input, bool useless)
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
                Weights = mask;
                return output;
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
                return output;
            }
        }

        public override Layer Init(bool isoutput)
        {
            return this;
        }

        public override void Descend(bool batchnorm)
        {
            throw new NotImplementedException();
        }

        public override void CalcGradients(List<double[]> inputs, Layer outputlayer)
        {
            throw new NotImplementedException();
        }
    }
}
