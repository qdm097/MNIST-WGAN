﻿using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms.Design;

namespace WGAN1
{
    class Maths
    {
        public static double ReLu(double number)
        {
            return (number > 0 ? number : 0);
        }
        public static double[] ReLu(double[] input)
        {
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = ReLu(input[i]);
            }
            return output;
        }
        public static List<double[]> ReLu(List<double[]> input)
        {
            List<double[]> output = new List<double[]>();
            for (int i = 0; i < input.Count; i++)
            {
                output.Add(ReLu(input[i]));
            }
            return output;
        }
        public static double ReLuDerriv(double number)
        {
            return (number >= 0 ? 1 : 0);
        }
        public static double[] ReLuDerriv(double[] input)
        {
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = ReLuDerriv(input[i]);
            }
            return output;
        }
        public static List<double[]> ReLuDerriv(List<double[]> input)
        {
            List<double[]> output = new List<double[]>();
            for (int i = 0; i < input.Count; i++)
            {
                output.Add(ReLuDerriv(input[i]));
            }
            return output;
        }
        public static double[] Tanh(double[] input)
        {
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Tanh(input[i]);
            }
            return output;
        }
        public static double[] TanhDerriv(double[] input)
        {
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = TanhDerriv(input[i]);
            }
            return output;
        }
        public static double Tanh(double number)
        {
            double x = (Math.Pow(Math.E, 2 * number) - 1) / (Math.Pow(Math.E, 2 * number) + 1);
            if (x == 0  || x is double.NaN || double.IsInfinity(x)) { return number; }
            else { return x; }
        }
        public static double TanhDerriv(double number)
        {
            return 1 - Math.Pow(Tanh(number), 2);
        }
        public static List<double[]> TanhDerriv(List<double[]> input)
        {
            List<double[]> output = new List<double[]>();
            for (int i = 0; i < input.Count; i++)
            {
                output.Add(TanhDerriv(input[i]));
            }
            return output;
        }
        public static List<double[]> Tanh(List<double[]> input)
        {
            List<double[]> output = new List<double[]>();
            for (int i = 0; i < input.Count; i++)
            {
                output.Add(Tanh(input[i]));
            }
            return output;
        }
        public static double[] Rescale(double[] array, double mean, double stddev)
        {
            //zscore
            double[] output = new double[array.Length];
            //var arraymean = CalcMean(array);
            //var output = Normalize(array, arraymean, CalcStdDev(array, arraymean));
            //Rescale the dataset (opposite of zscore)
            for (int i = 0; i < array.Length; i++)
            {
                // min + ((array[i] - setmin) * (max - min) / (setmax - setmin))
                output[i] = (array[i] * stddev) + mean;
            }
            return output;
        }
        /// <summary>
        /// Return an array of size latentsize of Gaussian distributed random variables (Box-Muller Transform)
        /// </summary>
        /// <param name="latentsize">The square root of the size of the latent space</param>
        /// <returns></returns>
        public static double[] RandomGaussian(Random r, int latentsize)
        {
            var latentspace = new double[latentsize];
            for (int i = 0; i < latentsize; i++)
            {
                double u1 = 1.0 - r.NextDouble(); //Uniform(0,1] random doubles
                double u2 = 1.0 - r.NextDouble();
                //Unscaled
                latentspace[i] = Math.Sqrt(-2.0 * Math.Log(u1)) *
                             Math.Sin(2.0 * Math.PI * u2); //Random normal(0,1)
            }
            return latentspace;
        }
        public static T[] Convert<T>(T[,] input)
        {
            T[] output = new T[input.Length];
            int iterator = 0;
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int ii = 0; ii < input.GetLength(1); ii++)
                {
                    output[iterator] = input[i, ii]; iterator++;
                }
            }
            return output;
        }
        public static T[,] Convert<T>(T[] input)
        {
            double sqrt = Math.Sqrt(input.Length);
            //If the input cannot be turned into a square array, throw an error
            if (sqrt != (int)sqrt) { throw new Exception("Invalid input array size"); }
            T[,] output = new T[(int)sqrt, (int)sqrt];
            int iterator = 0;
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int ii = 0; ii < output.GetLength(1); ii++)
                {
                    output[i, ii] = input[iterator]; iterator++;
                }
            }
            return output;
        }
        public static double[] Normalize(double[] array, double mean, double stddev)
        {
            //Prevent errors
            if (stddev == 0) { stddev = .000001; }
            double[] output = new double[array.Length];
            //Calc zscore
            for (int i = 0; i < array.Length; i++)
            {
                output[i] = (array[i] - mean) / stddev;
            }

            return output;
        }
        public static List<double[]> Normalize(List<double[]> inputs)
        {
            List<double[]> outputs = new List<double[]>();
            foreach (double[] d in inputs)
            {
                outputs.Add(Normalize(d));
            }
            return outputs;
        }
        public static double[] Normalize(double[] input)
        {
            double mean = 0;
            double stddev = 0;
            //Calc mean of data
            foreach (double d in input) { mean += d; }
            mean /= input.Length;
            //Calc std dev of data
            foreach (double d in input) { stddev += (d - mean) * (d - mean); }
            stddev /= input.Length;
            stddev = Math.Sqrt(stddev);
            //Prevent divide by zero b/c of sigma = 0
            if (stddev == 0) { stddev = .000001; }
            double[] output = new double[input.Length];
            //Calc zscores
            for (int i = 0; i < output.Length; i++)
            {
                output[i] = (input[i] - mean) / stddev;
            }

            return output;
        }
        public static double[,] Normalize(double[,] input)
        {
            double mean = 0;
            double stddev = 0;
            //Calc mean of data
            foreach (double d in input) { mean += d; }
            mean /= input.Length;
            //Calc std dev of data
            foreach (double d in input) { stddev += (d - mean) * (d - mean); }
            stddev /= input.Length;
            stddev = Math.Sqrt(stddev);
            //Prevent divide by zero b/c of sigma = 0
            if (stddev == 0) { stddev = .000001; }
            double[,] output = new double[input.GetLength(0), input.GetLength(1)];
            //Calc zscores
            for (int i = 0; i < output.GetLength(0); i++)
            {
                for (int ii = 0; ii < output.GetLength(1); ii++)
                {
                    output[i, ii] = (input[i, ii] - mean) / stddev;
                }
            }

            return output;
        }
        public static double[] Scale(double scale, double[] array)
        {
            double[] output = new double[array.Length];
            for (int i = 0; i < array.Length; i++)
            {
                output[i] = array[i] * scale;
            }
            return output;
        }
        public static double[,] Scale(double scale, double[,] array)
        {
            double[,] output = new double[array.GetLength(0), array.GetLength(1)];
            for (int i = 0; i < array.GetLength(0); i++)
            {
                for (int ii = 0; ii < array.GetLength(1); ii++)
                {
                    output[i, ii] = array[i, ii] * scale;
                }
            }
            return output;
        }
        /// <summary>
        /// Calculates the mean of the array
        /// </summary>
        /// <param name="array">The set from which the mean is taken</param>
        /// <returns></returns>
        public static double CalcMean(double[] array)
        {
            double mean = 0;
            foreach (double d in array) { mean += d; }
            mean /= array.Length;
            return mean;
        }
        /// <summary>
        /// Calculates the standard deviation of the array
        /// </summary>
        /// <param name="array">The set from which the stddev is taken</param>
        /// <param name="mean">The mean of the set</param>
        /// <returns></returns>
        public static double CalcStdDev(double[] array, double mean)
        {
            double stddev = 0;
            //Calc std dev of data
            foreach (double d in array) { stddev += ((d - mean) * (d - mean)); }
            stddev /= array.Length;
            stddev = Math.Sqrt(stddev);
            return stddev;
        }
        public static List<double[]> BatchNormalize(List<double[]> input)
        {
            double mean = 0, stddev = 0;
            foreach (double[] d in input) { double temp = CalcMean(d); mean += temp; stddev += CalcStdDev(d, temp); }
            return BatchNormalize(input, mean, stddev);
        }
        public static List<double[]> BatchNormalize(List<double[]> input, double mean, double stddev)
        {
            List<double[]> output = new List<double[]>();
            for (int i = 0; i < input.Count; i++)
            {
                output.Add(new double[input[i].Length]);
                for (int ii = 0; ii < input[i].Length; ii++)
                {
                    output[i][ii] = (input[i][ii] - mean) / stddev;
                }
            }
            return output;
        }
    }
}
