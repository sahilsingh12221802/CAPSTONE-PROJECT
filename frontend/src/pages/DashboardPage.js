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

  const speciesCounts = useMemo(() => {
    const counts = { Cattle: 0, Buffalo: 0, Unknown: 0 };
    records.forEach((item) => {
      if (item.label in counts) {
        counts[item.label] += 1;
      }
    });
    return counts;
  }, [records]);

  const breedCounts = useMemo(() => {
    const counts = {};
    records.forEach((item) => {
      if (item.breed) {
        counts[item.breed] = (counts[item.breed] || 0) + 1;
      }
    });
    return Object.entries(counts).sort((a, b) => b[1] - a[1]);
  }, [records]);

  const avgConfidence = useMemo(() => {
    if (!records.length) {
      return 0;
    }
    const total = records.reduce((acc, item) => acc + Number(item.confidence || 0), 0);
    return (total / records.length) * 100;
  }, [records]);

  const confidenceTrend = useMemo(() => {
    return records
      .slice(0, 12)
      .reverse()
      .map((item) => Number(item.confidence || 0));
  }, [records]);

  const chartPoints = useMemo(() => {
    if (!confidenceTrend.length) {
      return "";
    }

    return confidenceTrend
      .map((value, index) => {
        const x = (index / Math.max(confidenceTrend.length - 1, 1)) * 100;
        const y = 100 - value * 100;
        return `${x},${y}`;
      })
      .join(" ");
  }, [confidenceTrend]);

  const speciesEntries = useMemo(() => Object.entries(speciesCounts), [speciesCounts]);
  const speciesMax = useMemo(() => Math.max(1, ...speciesEntries.map(([, value]) => value)), [speciesEntries]);
  const breedMax = useMemo(() => Math.max(1, ...breedCounts.map(([, value]) => value)), [breedCounts]);

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-3xl font-bold text-gray-900">Project Dashboard</h2>
        <p className="mt-2 text-gray-600">Live overview of classification usage and outcomes.</p>
      </div>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard title="Total Uploads" value={records.length} subtitle="From MongoDB records" />
        <StatCard title="Cattle Predictions" value={speciesCounts.Cattle} />
        <StatCard title="Buffalo Predictions" value={speciesCounts.Buffalo} />
        <StatCard title="Avg Confidence" value={`${avgConfidence.toFixed(1)}%`} subtitle="Rolling dashboard average" />
      </section>

      {error ? <p className="rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p> : null}

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="card p-6">
          <h3 className="text-xl font-semibold text-gray-900">Species Distribution</h3>
          <div className="mt-4 space-y-3">
            {speciesEntries.map(([label, value]) => (
              <div key={label}>
                <div className="mb-1 flex items-center justify-between text-sm text-gray-700">
                  <span>{label}</span>
                  <span>{value}</span>
                </div>
                <div className="h-2 rounded-full bg-gray-100">
                  <div
                    className="h-2 rounded-full bg-emerald-600"
                    style={{ width: `${Math.max((value / speciesMax) * 100, value > 0 ? 6 : 0)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card p-6">
          <h3 className="text-xl font-semibold text-gray-900">Breed Distribution</h3>
          {breedCounts.length ? (
            <div className="mt-4 space-y-3">
              {breedCounts.map(([label, value]) => (
                <div key={label}>
                  <div className="mb-1 flex items-center justify-between text-sm text-gray-700">
                    <span>{label.replaceAll("_", " ")}</span>
                    <span>{value}</span>
                  </div>
                  <div className="h-2 rounded-full bg-gray-100">
                    <div
                      className="h-2 rounded-full bg-indigo-600"
                      style={{ width: `${Math.max((value / breedMax) * 100, 8)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-sm text-gray-500">No breed-level records available yet.</p>
          )}
        </div>

        <div className="card p-6 lg:col-span-2">
          <h3 className="text-xl font-semibold text-gray-900">Confidence Trend (Last 12)</h3>
          {confidenceTrend.length ? (
            <div className="mt-4 rounded-xl border border-gray-200 bg-gray-50 p-3">
              <svg viewBox="0 0 100 100" className="h-44 w-full">
                <polyline
                  fill="none"
                  stroke="#059669"
                  strokeWidth="2"
                  points={chartPoints}
                  vectorEffect="non-scaling-stroke"
                />
              </svg>
              <p className="text-xs text-gray-500">Left is oldest, right is newest classification.</p>
            </div>
          ) : (
            <p className="mt-3 text-sm text-gray-500">Add some classifications to see confidence trend.</p>
          )}
        </div>
      </section>

      <section className="card p-6">
        <h3 className="text-xl font-semibold text-gray-900">Recent Classifications</h3>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-left text-sm">
            <thead>
              <tr className="text-xs uppercase tracking-wide text-gray-500">
                <th className="px-3 py-2">Label</th>
                <th className="px-3 py-2">Breed</th>
                <th className="px-3 py-2">Confidence</th>
                <th className="px-3 py-2">Filename</th>
                <th className="px-3 py-2">Timestamp</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {records.slice(0, 8).map((item) => (
                <tr key={item.id}>
                  <td className="px-3 py-2 font-medium text-gray-900">{item.label}</td>
                  <td className="px-3 py-2 text-gray-700">{item.breed ? item.breed.replaceAll("_", " ") : "-"}</td>
                  <td className="px-3 py-2 text-gray-700">{(item.confidence * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2 text-gray-700">{item.filename}</td>
                  <td className="px-3 py-2 text-gray-500">{new Date(item.timestamp).toLocaleString()}</td>
                </tr>
              ))}
              {records.length === 0 ? (
                <tr>
                  <td className="px-3 py-4 text-gray-500" colSpan={5}>
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
