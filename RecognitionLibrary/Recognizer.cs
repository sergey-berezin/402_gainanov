using System;
using System.IO;
using System.Linq;
using System.Drawing;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Collections.Concurrent;
using System.Diagnostics;
using Microsoft.ML;
using Microsoft.ML.OnnxRuntime;
using static Microsoft.ML.Transforms.Image.ImageResizingEstimator;

namespace RecognitionLibrary
{
    public class Recognizer
    {
        const string modelPath = @"C:\Users\Gaynanov_D\projects\yolov4.onnx";

        static readonly string[] classesNames = new string[] {
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
            "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
            "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        static readonly Dictionary<string, int[]> shapeDictionary = new Dictionary<string, int[]>
        {
             { "input_1:0", new[] { 1, 416, 416, 3 } },
             { "Identity:0", new[] { 1, 52, 52, 3, 85 } },
             { "Identity_1:0", new[] { 1, 26, 26, 3, 85 } },
             { "Identity_2:0", new[] { 1, 13, 13, 3, 85 } },
        };

        static readonly string[] inputColumnNames = new string[] { "input_1:0" };

        static readonly string[] outputColumnNames = new string[]
        {
            "Identity:0",
            "Identity_1:0",
            "Identity_2:0"
        };

        public static System.Action Recognition_task(string filename,
            CancellationTokenSource token,
            ConcurrentQueue<Tuple<string, YoloV4Result>> detectionResults,
            MLContext mlContext,
            Microsoft.ML.Data.TransformerChain<Microsoft.ML.Transforms.Onnx.OnnxTransformer> model)
        {
            return () =>
            {
                if (token.IsCancellationRequested)
                {
                    return;
                }

                var bitmap = new Bitmap(Image.FromFile(filename));
                var predict = mlContext.Model.CreatePredictionEngine<YoloV4BitmapData, YoloV4Prediction>(model).Predict(new YoloV4BitmapData() { Image = bitmap });
                var results = predict.GetResults(classesNames, 0.3f, 0.7f);

                foreach (var detected in results)
                {
                    if (token.IsCancellationRequested)
                    {
                        return;
                    }

                    var resTuple = new Tuple<string, YoloV4Result>(filename, detected);
                    detectionResults.Enqueue(resTuple);
                }
            };
        }

        public static void Detect(string directory,
            CancellationTokenSource token,
            ConcurrentQueue<Tuple<string, YoloV4Result>> detectionResults)
        {
            MLContext mlContext = new MLContext();

            var pipeline = mlContext.Transforms.ResizeImages(inputColumnName: "bitmap", outputColumnName: "input_1:0", imageWidth: 416, imageHeight: 416, resizing: ResizingKind.IsoPad)
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input_1:0", scaleImage: 1f / 255f, interleavePixelColors: true))
                .Append(mlContext.Transforms.ApplyOnnxModel(
                    shapeDictionary: shapeDictionary,
                    inputColumnNames: inputColumnNames,
                    outputColumnNames: outputColumnNames,
                    modelFile: modelPath, recursionLimit: 100));

            Microsoft.ML.Data.TransformerChain<Microsoft.ML.Transforms.Onnx.OnnxTransformer> model = pipeline.Fit(mlContext.Data.LoadFromEnumerable(new List<YoloV4BitmapData>()));

            var filenames = Directory.GetFiles(directory).Select(path => Path.GetFullPath(path)).ToArray();
            var tasks = new List<Task>();
            foreach (string filename in filenames)
            {
                if (!filename.Contains(".jpg"))
                {
                    continue;
                }
                tasks.Add(Task.Factory.StartNew(Recognition_task(filename, token, detectionResults, mlContext, model), token.Token));
            }
            Task.WaitAll(tasks.ToArray());
        }
    }
}
