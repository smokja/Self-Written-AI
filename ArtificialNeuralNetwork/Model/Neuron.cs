using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ArtificialNeuralNetwork.Model
{
    public class Neuron
    {
        public List<double> Weights { get; set; }
        public double Value { get; set; }
        public Neuron()
        {
            Weights = new List<double>();
        }
    }
}
