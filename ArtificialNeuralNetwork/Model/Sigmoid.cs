using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork.Model
{
    public static class Sigmoid
    {
        public static double Process(double input)
        {
            return 1 / (1 + Math.Exp(-input));
        }
        public static double Derivative(double input)
        {
         
            return input * (1 - input);
        }
    }
}
