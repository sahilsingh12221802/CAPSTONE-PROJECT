import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import BreedDetailsPage from "./pages/BreedDetailsPage";
import BreedsPage from "./pages/BreedsPage";
import ClassifyPage from "./pages/ClassifyPage";
import DashboardPage from "./pages/DashboardPage";
import HomePage from "./pages/HomePage";

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/breeds" element={<BreedsPage />} />
        <Route path="/breeds/:id" element={<BreedDetailsPage />} />
        <Route path="/classify" element={<ClassifyPage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default App;
