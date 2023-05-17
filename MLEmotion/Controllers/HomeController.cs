using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using MLEmotion.Models;
using System.Diagnostics;
using System.Drawing;
using static Microsoft.ML.DataOperationsCatalog;
using static System.Net.Mime.MediaTypeNames;
using MLEmotion.Models;


namespace MLEmotion.Controllers
{
    public class HomeController : Controller
    {

        private readonly ILogger<HomeController> _logger;
        private readonly IWebHostEnvironment _env;

        public HomeController(ILogger<HomeController> logger, IWebHostEnvironment env)
        {
            _logger = logger;
            _env = env;
        }

        public IActionResult Index()
        {

            string contentRootPath = _env.ContentRootPath;
            string imagePath_fromfolder = Path.Combine(contentRootPath, "wwwroot", "testimage", "daod.jpg");

            Trainmodel(imagePath_fromfolder);


            return View();
        }

        public IActionResult Privacy()
        {
            return View();
        }

        public void Trainmodel(string imagePath)
        {
            // Create a new MLContext
            var mlContext = new MLContext();

            // Converting image to model input
            var input = ConvertImageToInput(imagePath);

            // Load data
            var data = new[] { new ImageData { Conv2DInput = input } };
            var dataView = mlContext.Data.LoadFromEnumerable(data);

            // Define pipeline
            string contentRootPath = _env.ContentRootPath;
            string modelPath_fromfolder = Path.Combine(contentRootPath, "wwwroot", "model", "my_model.onnx");

            var pipeline = mlContext.Transforms.ApplyOnnxModel(
                outputColumnNames: new[] { "dense_1" },
                inputColumnNames: new[] { "conv2d_input" },
                modelFile: modelPath_fromfolder
            );
           
            // Fit the pipeline to the data
            var model = pipeline.Fit(dataView);

            // Create a prediction engine
             var predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, EmotionPrediction>(model);

            // Use the model to predict the output of the sample data
             EmotionPrediction prediction = predictionEngine.Predict(new ImageData { Conv2DInput = input });

            var predictedLabelIndex = prediction.PredictedLabels
            .Select((value, index) => new { Value = value, Index = index })
            .Aggregate((a, b) => (a.Value > b.Value) ? a : b)
            .Index;

            ViewBag.PredictedLabel = (EmotionLable)predictedLabelIndex; 

        }

        //onvert an image into a grayscale and resize it to 48x48
        private float[] ConvertImageToInput(string imagePath)
        {
            using var bitmap = new Bitmap(System.Drawing.Image.FromFile(imagePath), new Size(48, 48));
            var input = new float[48 * 48];
            for (int y = 0; y < bitmap.Height; y++)
            {
                for (int x = 0; x < bitmap.Width; x++)
                {
                    var color = bitmap.GetPixel(x, y);
                    var grayScale = (color.R * 0.3) + (color.G * 0.59) + (color.B * 0.11);
                    input[y * bitmap.Width + x] = (float)grayScale / 255;
                }
            }
            return input;
        }


        public class ImageData
        {
            [ColumnName("conv2d_input")]
            [VectorType(1, 48, 48, 1)] // Adjust according to your ONNX model input shape
            public float[] Conv2DInput { get; set; }
        }

        public class EmotionPrediction
        {
            [ColumnName("dense_1")] // Adjust according to your ONNX model output tensor name
            [VectorType(7)] // If your model outputs a probability distribution over 7 classes, adjust if not correct
            public float[] PredictedLabels { get; set; }
        }



        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}