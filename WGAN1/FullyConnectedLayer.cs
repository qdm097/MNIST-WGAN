﻿using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class FullyConnectedLayer : Layer
    {
        double[,] WRMSGrad { get; set; }
        double[,] WeightGradient { get; set; }
        double[,] WUpdates { get; set; }
        //Biases
        public double[] Biases { get; set; }
        double[] BRMSGrad { get; set; }
        double[] BiasGradient { get; set; }
        double[] BUpdates { get; set; }

        public double AvgGradient { get; set; }
        public FullyConnectedLayer(int l, int il)
        {
            Length = l; InputLength = il;
            WeightGradient = new double[l, il];
            WRMSGrad = new double[l, il];
            BiasGradient = new double[l];
            BRMSGrad = new double[l];
            Weights = new double[l, il];
            Biases = new double[l];
        }
        public override Layer Init(bool isoutput)
        {
            var r = new Random();
            //All layers have weights
            Weights = new double[Length, InputLength];
            //Output layer has no biases
            if (!isoutput) { Biases = new double[Length]; }
            //Initialize weights (and biases to zero)
            for (int j = 0; j < Length; j++)
            {
                for (int jj = 0; jj < InputLength; jj++)
                {
                    //LeCun initialization
                    Weights[j, jj] = (r.NextDouble() > .5 ? -1 : 1) * r.NextDouble() * Math.Sqrt(3d / (InputLength * InputLength));
                }
            }
            return this;
        }
        /// <summary>
        /// Applies the gradients to the weights as a batch
        /// </summary>
        /// <param name="batchsize">The number of trials run per cycle</param>
        /// <param name="clipparameter">What the max/min </param>
        /// <param name="RMSDecay">How quickly the RMS gradients decay</param>
        public override void Descend()
        {
            //Calculate gradients
            WUpdates = new double[Length, InputLength];
            BUpdates = new double[Length];

            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Normal gradient descent update
                    WUpdates[i, ii] = WeightGradient[i, ii] * (2d / NN.BatchSize);
                    //Root mean square propegation
                    if (NN.UseRMSProp)
                    {
                        WRMSGrad[i, ii] = (WRMSGrad[i, ii] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (WUpdates[i, ii] * WUpdates[i, ii]));
                        WUpdates[i, ii] = (WUpdates[i, ii] / (Math.Sqrt(WRMSGrad[i, ii])/* + NN.Infinitesimal*/));
                    }
                    WUpdates[i, ii] *= NN.LearningRate;
                }
                //Normal gradient descent update
                BUpdates[i] = BiasGradient[i] * (2d / NN.BatchSize);
                //Root mean square propegation
                if (NN.UseRMSProp)
                {
                    BRMSGrad[i] = (BRMSGrad[i] * NN.RMSDecay) + ((1 - NN.RMSDecay) * (BUpdates[i] * BUpdates[i]));
                    BUpdates[i] = (BUpdates[i] / (Math.Sqrt(BRMSGrad[i])/* + NN.Infinitesimal*/));
                }
                BUpdates[i] *= NN.LearningRate;
            }
            //Gradient normalization
            if (NN.NormGradients) 
            { 
                WUpdates = Maths.Scale(NN.LearningRate, Maths.Normalize(WUpdates)); 
                BUpdates = Maths.Scale(NN.LearningRate, Maths.Normalize(BUpdates)); 
            }
            //Apply updates
            for (int i = 0; i < Length; i++)
            {
                for (int ii = 0; ii < InputLength; ii++)
                {
                    //Update weight and average
                    Weights[i, ii] -= WUpdates[i, ii];
                    AvgGradient -= WUpdates[i, ii];
                    //Weight clipping
                    if (NN.UseClipping)
                    {
                        if (Weights[i, ii] > NN.ClipParameter) { Weights[i, ii] = NN.ClipParameter; }
                        if (Weights[i, ii] < -NN.ClipParameter) { Weights[i, ii] = -NN.ClipParameter; }
                    }
                }
                Biases[i] -= BUpdates[i];
                //Bias clipping
                if (NN.UseClipping)
                {
                    if (Biases[i] > NN.ClipParameter) { Biases[i] = NN.ClipParameter; }
                    if (Biases[i] < -NN.ClipParameter) { Biases[i] = -NN.ClipParameter; }
                }
            }

            //Reset gradients
            WeightGradient = new double[Length, InputLength];
            BiasGradient = new double[Length];
        }
        /// <summary>
        /// Descent for other layers
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="isoutput">Whether the layer is the output layer</param>
        /// <summary>
        /// Backpropegation of error and calcluation of gradients
        /// </summary>
        /// <param name="input">Previous layer's values (only needed if calculating gradients)</param>
        /// <param name="isoutput">Whether the layer is the output layer</param>
        public override void CalcGradients(List<double[]> inputs, Layer outputlayer)
        {
            for (int b = 0; b < NN.BatchSize; b++)
            {
                //Calculate gradients
                for (int i = 0; i < Length; i++)
                {
                    double zval = ZVals[b][i];
                    if (ActivationFunction == 0) { zval = Maths.TanhDerriv(ZVals[b][i]); }
                    if (ActivationFunction == 1) { zval = Maths.ReLuDerriv(ZVals[b][i]); }

                    for (int ii = 0; ii < InputLength; ii++)
                    {
                        //Weight gradients
                        WeightGradient[i, ii] += inputs[b][ii] * zval * Errors[b][i];
                    }
                    if (outputlayer is null) { continue; }
                    //Bias gradients
                    BiasGradient[i] += zval * Errors[b][i];
                }
            }
        }
        public override void Calculate(List<double[]> inputs, bool output)
        {
            ZVals = new List<double[]>();
            for (int b = 0; b < NN.BatchSize; b++)
            {
                var vals = new double[Length];
                for (int k = 0; k < Length; k++)
                {
                    //Values = (weights * inputs) + biases
                    for (int j = 0; j < InputLength; j++)
                    {
                        vals[k] += Weights[k, j] * inputs[b][j];
                    }
                    //Output layers don't use biases
                    if (!output)
                    {
                        vals[k] += Biases[k];
                    }
                }
                ZVals.Add(vals);
            }
            //If normalizing, do so, but only if it won't return an all-zero matrix
            if (NN.NormOutputs && ZVals[0].Length > 1) { ZVals = Maths.Normalize(ZVals); }
            //Use the specified type of activation function
            if (ActivationFunction == 0) { Values = Maths.Tanh(ZVals); return; }
            if (ActivationFunction == 1) { Values = Maths.ReLu(ZVals); return; }
            Values = ZVals;
        }
    }
}
