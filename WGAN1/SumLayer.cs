﻿using System;
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
        /// <summary>
        /// Adds two matrices together
        /// </summary>
        /// <param name="inputs1">Matrix 1</param>
        /// <param name="inputs2">Matrix 2</param>
        public void Calculate(List<double[]> inputs1, List<double[]> inputs2)
        {
            ZVals = new List<double[]>();
            if (inputs1.Count != inputs2.Count) { throw new Exception("List sizes do not match"); }
            for (int b = 0; b < NN.BatchSize; b++)
            {
                if (inputs1[b].Length != inputs2[b].Length)
                {
                    throw new Exception("Array sizes do not match");
                }

                double[] output = new double[inputs1[b].Length];
                for (int i = 0; i < inputs1[b].Length; i++)
                {
                    output[i] = inputs1[b][i] + inputs2[b][i];
                }
                ZVals.Add(output);
            }
            //If normalizing, do so, but only if it won't return an all-zero matrix
            if (NN.NormOutputs && ZVals[0].Length > 1) { ZVals = Maths.Normalize(ZVals); }
            //Use the specified type of activation function
            if (ActivationFunction == 0) { Values = Maths.Tanh(ZVals); return; }
            if (ActivationFunction == 1) { Values = Maths.ReLu(ZVals); return; }
            else { Values = ZVals; }
        }

        public override void Descend()
        {
            throw new NotImplementedException();
        }

        public override Layer Init(bool b) { return this; }
    }
}
