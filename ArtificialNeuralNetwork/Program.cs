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
            Model.ArtificialNeuralNetwork bestNetwork = null;
            double lowestError = 4; // highest error is 4
            for (int i = 0; i < 5000000; i++) {
                double currentError = 0;
                var localNetwork = new Model.ArtificialNeuralNetwork(2, 1, 2, new int[2] { 3, 4 });
                
                var output = localNetwork.FeedForward(new double[2] { 1, 0 });
                currentError += 1 - output[0];
                
                output = localNetwork.FeedForward(new double[2] { 0, 1 });
                currentError += 1 - output[0]; 

                output = localNetwork.FeedForward(new double[2] { 1, 1 });
                currentError += 0 + output[0]; 

                output = localNetwork.FeedForward(new double[2] { 0, 0 });
                currentError += 0 + output[0]; 
                
                if (currentError < lowestError) {
                    bestNetwork = localNetwork;
                    lowestError = currentError;
                }
            }

            var bestOutput = bestNetwork.FeedForward(new double[2] { 1, 1 });
            bestOutput.ForEach(x => Console.WriteLine(x));
            bestOutput = bestNetwork.FeedForward(new double[2] { 0, 0 });
            bestOutput.ForEach(x => Console.WriteLine(x));
            bestOutput = bestNetwork.FeedForward(new double[2] { 1, 0 });
            bestOutput.ForEach(x => Console.WriteLine(x));
            bestOutput = bestNetwork.FeedForward(new double[2] { 0, 1 });
            bestOutput.ForEach(x => Console.WriteLine(x));
        }
    }
}
