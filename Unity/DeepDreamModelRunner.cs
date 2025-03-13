using UnityEngine;
// Assume Unity Sentis has similar namespaces to Barracuda – adjust based on your Sentis version.
using Unity.Sentis;

public class DeepDreamModelRunner : MonoBehaviour
{
    [Header("Model & Input")]
    [Tooltip("Exported ONNX model file.")]
    public TextAsset onnxModelFile; // Set this in the Inspector
    [Tooltip("Input texture to process.")]
    public Texture2D inputTexture; // Set this in the Inspector

    [Header("Display")]
    [Tooltip("Mesh renderer where the output texture will be shown.")]
    public MeshRenderer targetRenderer; // Assign a mesh renderer with a material that uses the provided shader

    // Internal variables for the model and inference worker
    private Model runtimeModel;
    private IWorker worker;

    // Toggle for running the model
    private bool isRunning = false;

    // For timing inference (optional)
    private float lastInferenceTime = 0f;

    void Start()
    {
        // Load the ONNX model (ensure the correct Sentis API is used)
        runtimeModel = ModelLoader.Load(onnxModelFile);
        worker = WorkerFactory.CreateWorker(runtimeModel);
        Debug.Log("ONNX model loaded and worker created.");
    }

    void Update()
    {
        if (!isRunning)
            return;

        // Optional: Start timing the inference
        float startTime = Time.realtimeSinceStartup;

        // Convert the input texture to a tensor.
        // This assumes that the Sentis API has a Tensor constructor that accepts a Texture2D.
        Tensor inputTensor = new Tensor(inputTexture, channels: 3);

        // Execute the model inference.
        worker.Execute(inputTensor);

        // Get the output tensor (assuming the model returns a single output named "output").
        Tensor outputTensor = worker.PeekOutput();

        // Convert the output tensor to a Texture2D.
        Texture2D outputTexture = new Texture2D(inputTexture.width, inputTexture.height, TextureFormat.RGBA32, false);
        Color[] colors = new Color[inputTexture.width * inputTexture.height];

        // Assuming outputTensor shape is [1, 3, height, width] in RGB order.
        for (int h = 0; h < inputTexture.height; h++)
        {
            for (int w = 0; w < inputTexture.width; w++)
            {
                float r = outputTensor[0, 0, h, w];
                float g = outputTensor[0, 1, h, w];
                float b = outputTensor[0, 2, h, w];
                colors[h * inputTexture.width + w] = new Color(r, g, b, 1.0f);
            }
        }
        outputTexture.SetPixels(colors);
        outputTexture.Apply();

        // Assign the generated texture to the material of the target mesh.
        targetRenderer.material.SetTexture("_MainTex", outputTexture);

        // Log output info and timing.
        lastInferenceTime = Time.realtimeSinceStartup - startTime;
        Debug.Log("Inference complete. Time taken: " + lastInferenceTime.ToString("F4") + " seconds.");

        // Dispose of tensors to free resources.
        inputTensor.Dispose();
        outputTensor.Dispose();
    }

    // Display a simple on-screen GUI button to start/stop the model.
    void OnGUI()
    {
        Rect buttonRect = new Rect(10, 10, 150, 50);
        string buttonText = isRunning ? "Stop Model" : "Start Model";
        if (GUI.Button(buttonRect, buttonText))
        {
            isRunning = !isRunning;
            Debug.Log("Model running state toggled: " + isRunning);
        }
    }

    void OnDestroy()
    {
        if(worker != null)
            worker.Dispose();
    }
}
