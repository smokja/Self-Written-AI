using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ArtificialNeuralNetwork.Model;

namespace ArtificialNeuralNetwork
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var network = new Model.ArtificialNeuralNetwork(2, 1, 0, new int[0]);
            var output = network.FeedForward(new double[2] { 20.5, -20.5 });
            output.ForEach(x => Console.WriteLine(x));
            Console.ReadKey();
        }
    }
}
