using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace WGAN1
{
    interface iLayer
    {
        double[,] Weights { get; set; }
        double[] Values { get; set; }
        double[] Errors { get; set; }
        int Length { get; set; }
        int InputLength { get; set; }

        iLayer Init(bool isoutput);
        void Descend(int batchsize, double learningrate, double clipparameter, double RMSDecay);
        /// <summary>
        /// Descent for the first layer
        /// </summary>
        /// <param name="input">Original image (optionally normalized)</param>
        void Descend(double[,] input);
        /// <summary>
        /// Descent for other layers
        /// </summary>
        /// <param name="input">Previous layer's values</param>
        /// <param name="output">Whether the layer is the output layer</param>
        void Descend(double[] input, bool output);
        void Backprop(iLayer output);
        void Backprop(double correct);
        void Calculate(double[] input, bool output);
        void Calculate(double[,] input, bool output);
    }
}
