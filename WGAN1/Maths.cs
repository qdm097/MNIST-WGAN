using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    class Maths
    {
        public static double[] Tanh(double[] input)
        {
            var output = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = Tanh(input[i]);
            }
            return output;
        }
        public static double[,] Tanh(double[,] input)
        {
            var output = new double[input.GetLength(0), input.GetLength(1)];
            for (int i = 0; i < input.GetLength(0); i++)
            {
                for (int ii = 0; ii < input.GetLength(1); ii++)
                {
                    output[i, ii] = Tanh(input[i, ii]);
                }
            }
            return output;
        }
        public static double Tanh(double number)
        {
            return (Math.Pow(Math.E, 2 * number) - 1) / (Math.Pow(Math.E, 2 * number) + 1);
        }
        public static double TanhDerriv(double number)
        {
            return (1 - number) * (1 + number);
        }
        public static double[] Rescale(double[] array, int min, int max)
        {
            double setmin = 0, setmax = 0;
            //Find the minimum and maximum values of the dataset
            foreach(double d in array)
            {
                if (d > setmax) { setmax = d; }
                if (d < setmin) { setmin = d; }
            }
            //Rescale the dataset
            for (int i = 0; i < array.Length; i++)
            {
                // min + ((array[i] - setmin) * (max - min) / (setmax - setmin))
                array[i] = 255 * ((array[i] - setmin) / (setmax - setmin));
            }
            return array;
        }
        /// <summary>
        /// Return an array of size latentsize of Gaussian distributed random variables (Box-Muller Transform)
        /// </summary>
        /// <param name="latentsize">The square root of the size of the latent space</param>
        /// <returns></returns>
        public static double[] RandomGaussian(int latentsize)
        {
            var latentspace = new double[latentsize];
            Random rand = new Random(); //Reuse this if you are generating many
            for (int i = 0; i < latentsize; i++)
            {
                double u1 = 1.0 - rand.NextDouble(); //Uniform(0,1] random doubles
                double u2 = 1.0 - rand.NextDouble();
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
        public static double[] Normalize(double[] array)
        {
            double mean = 0;
            double stddev = 0;
            //Calc mean of data
            foreach (double d in array) { mean += d; }
            mean /= array.Length;
            //Calc std dev of data
            foreach (double d in array) { stddev += (d - mean) * (d - mean); }
            stddev /= array.Length;
            stddev = Math.Sqrt(stddev);
            //Prevent divide by zero b/c of sigma = 0
            if (stddev == 0) { stddev = .000001; }
            //Calc zscore
            for (int i = 0; i < array.Length; i++)
            {
                array[i] = (array[i] - mean) / stddev;
            }

            return array;
        }
        public static double[,] Normalize(double[,] array, int depth, int count)
        {
            double[] smallarray = new double[depth * count];
            int iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count; ii++)
                {
                    smallarray[iterator] = array[i, ii];
                    iterator++;
                }
            }
            smallarray = Normalize(smallarray);
            iterator = 0;
            for (int i = 0; i < depth; i++)
            {
                for (int ii = 0; ii < count; ii++)
                {
                    array[i, ii] = smallarray[iterator];
                    iterator++;
                }
            }
            return array;
        }
    }
}
