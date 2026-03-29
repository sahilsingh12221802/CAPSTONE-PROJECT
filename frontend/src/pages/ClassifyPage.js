import axios from "axios";
import { LoaderCircle, UploadCloud } from "lucide-react";
import { useState } from "react";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

function ClassifyPage() {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onSubmit = async (event) => {
    event.preventDefault();

    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await axios.post(`${API_BASE_URL}/classify`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (response.data?.success) {
        setResult(response.data.data);
      } else {
        setError("Unexpected API response.");
      }
    } catch (requestError) {
      setError(requestError?.response?.data?.detail || "Network error. Is backend running on port 8000?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-[3fr,2fr]">
      <section className="card p-7">
        <h2 className="text-2xl font-bold text-gray-900">Classification</h2>
        <p className="mt-2 text-sm text-gray-600">
          Upload an animal image and run inference through your trained model.
        </p>

        <form onSubmit={onSubmit} className="mt-6 space-y-4">
          <label className="flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed border-gray-300 bg-gray-50 px-4 py-10 text-center transition hover:border-emerald-500 hover:bg-emerald-50">
            <UploadCloud className="mb-2 text-emerald-600" size={28} />
            <span className="text-sm font-medium text-gray-700">
              {file ? file.name : "Click to select image"}
            </span>
            <span className="mt-1 text-xs text-gray-500">PNG, JPG, JPEG supported</span>
            <input type="file" accept="image/*" onChange={(event) => setFile(event.target.files?.[0] || null)} className="hidden" />
          </label>

          <button
            type="submit"
            disabled={loading}
            className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-600 px-4 py-3 text-sm font-semibold text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-300"
          >
            {loading ? <LoaderCircle size={18} className="animate-spin" /> : null}
            {loading ? "Classifying..." : "Run Classification"}
          </button>

          {error ? <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}
        </form>
      </section>

      <aside className="card p-7">
        <h3 className="text-lg font-semibold text-gray-900">Result</h3>

        {!result && !loading ? (
          <p className="mt-4 text-sm text-gray-600">No prediction yet. Upload an image and run classification.</p>
        ) : null}

        {loading ? (
          <div className="mt-5 inline-flex items-center gap-2 text-sm text-gray-600">
            <LoaderCircle size={18} className="animate-spin text-emerald-600" />
            <span>Model is processing your image...</span>
          </div>
        ) : null}

        {result ? (
          <div
            className={`mt-5 space-y-3 rounded-xl border p-4 ${
              result.label === "Unknown"
                ? "border-amber-200 bg-amber-50"
                : "border-emerald-100 bg-emerald-50"
            }`}
          >
            <div>
              <p
                className={`text-xs uppercase tracking-[0.14em] ${
                  result.label === "Unknown" ? "text-amber-700" : "text-emerald-700"
                }`}
              >
                Prediction
              </p>
              <p className="text-2xl font-bold text-gray-900">{result.label}</p>
            </div>
            <div>
              <p
                className={`text-xs uppercase tracking-[0.14em] ${
                  result.label === "Unknown" ? "text-amber-700" : "text-emerald-700"
                }`}
              >
                Confidence
              </p>
              <p className="text-lg font-semibold text-gray-900">{(result.confidence * 100).toFixed(2)}%</p>
            </div>
            {result.message ? <p className="text-sm text-gray-700">{result.message}</p> : null}
            <p className="text-xs text-gray-500">File: {result.filename}</p>
          </div>
        ) : null}
      </aside>
    </div>
  );
}

export default ClassifyPage;
