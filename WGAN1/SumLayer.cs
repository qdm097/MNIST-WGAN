using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class SumLayer : Layer
    {
        public SumLayer(int l, int il)
        {
            Length = l; 
            InputLength = il;
        }

        public override void CalcGradients(List<double[]> inputs, Layer outputlayer)
        {
            throw new NotImplementedException();
        }

        public override void Calculate(List<double[]> input, bool output)
        {
            throw new NotImplementedException();
        }
        public void Calculate(List<double[]> inputs1, List<double[]> inputs2)
        {
            ZVals = new List<double[]>();
            if (inputs1.Count != inputs2.Count) { throw new Exception("List sizes do not match"); }
            for (int j = 0; j < inputs1.Count; j++)
            {
                if (inputs1[j].Length != inputs2[j].Length)
                {
                    throw new Exception("Array sizes do not match");
                }

                double[] output = new double[inputs1[j].Length];
                for (int i = 0; i < inputs1[j].Length; i++)
                {
                    output[i] = inputs1[j][i] + inputs2[j][i];
                }
                ZVals.Add(output);
            }
        }

        public override void Descend(int batchsize, bool batchnorm)
        {
            throw new NotImplementedException();
        }

        public override Layer Init(bool b) { return this; }
    }
}
