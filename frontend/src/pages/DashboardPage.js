import axios from "axios";
import { useEffect, useMemo, useState } from "react";
import StatCard from "../components/StatCard";

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || "http://localhost:8000";

function DashboardPage() {
  const [records, setRecords] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    const fetchRecords = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/records`);
        setRecords(response.data?.data || []);
      } catch {
        setError("Could not load dashboard records. Start backend server and refresh.");
      }
    };

    fetchRecords();
  }, []);

  const cattleCount = useMemo(() => records.filter((item) => item.label === "Cattle").length, [records]);
  const buffaloCount = useMemo(() => records.filter((item) => item.label === "Buffalo").length, [records]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Project Dashboard</h2>
        <p className="mt-2 text-gray-600">Live overview of classification usage and outcomes.</p>
      </div>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard title="Total Uploads" value={records.length} subtitle="From MongoDB records" />
        <StatCard title="Cattle Predictions" value={cattleCount} />
        <StatCard title="Buffalo Predictions" value={buffaloCount} />
        <StatCard title="Model Accuracy" value="89.5%" subtitle="Demo metric" />
      </section>

      {error ? <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}

      <section className="card p-6">
        <h3 className="text-xl font-semibold text-gray-900">Recent Classifications</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-left text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wide text-gray-500">
                <th className="px-3 py-2">Label</th>
                <th className="px-3 py-2">Confidence</th>
                <th className="px-3 py-2">Filename</th>
                <th className="px-3 py-2">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {records.slice(0, 8).map((item) => (
                <tr key={item.id}>
                  <td className="px-3 py-2 font-medium text-gray-900">{item.label}</td>
                  <td className="px-3 py-2 text-gray-700">{(item.confidence * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2 text-gray-700">{item.filename}</td>
                  <td className="px-3 py-2 text-gray-500">{new Date(item.timestamp).toLocaleString()}</td>
                </tr>
              ))}
              {records.length === 0 ? (
                <tr>
                  <td className="px-3 py-4 text-gray-500" colSpan={4}>
                    No records found.
                  </td>
                </tr>
              ) : null}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

export default DashboardPage;
