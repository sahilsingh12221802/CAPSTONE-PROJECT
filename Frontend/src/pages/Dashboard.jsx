import ImageUpload from "../components/ImageUpload"

const cattleBreeds = [
  {
    name: "Gir",
    desc: "Indigenous Indian cattle breed known for heat tolerance and quality milk production.",
  },
  {
    name: "Holstein Friesian",
    desc: "Exotic high-yield dairy breed widely used in organized dairy programs.",
  },
  {
    name: "Jersey",
    desc: "Compact dairy breed valued for high butterfat milk content.",
  },
  {
    name: "Sahiwal",
    desc: "Native Indian breed with excellent adaptability to tropical climates.",
  },
]

const buffaloBreeds = [
  {
    name: "Murrah",
    desc: "Elite buffalo breed known for high milk yield and fat percentage.",
  },
  {
    name: "Jaffrabadi",
    desc: "Heavy buffalo breed recognized for strength and productivity.",
  },
]

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-[#f5f7f4] text-gray-800">

      {/* ================= HERO ================= */}
      <section className="bg-gradient-to-br from-[#2f5d50] to-[#3f7f6a] text-white py-16 px-6">
        <div className="max-w-5xl mx-auto text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-wide">
            Animal Type Classification Dashboard
          </h1>
          <p className="text-green-100 max-w-3xl mx-auto">
            A smart image-based system for identifying cattle and buffalo breeds,
            supporting genetic improvement initiatives under the Rashtriya Gokul Mission.
          </p>
        </div>
      </section>

      {/* ================= CONTEXT STRIP ================= */}
      <section className="bg-white shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-6 text-center text-gray-600">
          This dashboard serves as the frontend interface for automated animal
          classification using AI and computer vision technologies.
        </div>
      </section>

      {/* ================= MAIN CONTENT ================= */}
      <main className="max-w-7xl mx-auto px-6 py-12 space-y-20">

        {/* ===== SYSTEM INSIGHTS ===== */}
        <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {[
            { label: "Total Breeds", value: "6" },
            { label: "Cattle Breeds", value: "4" },
            { label: "Buffalo Breeds", value: "2" },
            { label: "AI Model", value: "Under Development" },
          ].map((item) => (
            <div
              key={item.label}
              className="bg-white rounded-xl border border-gray-200 p-6 text-center"
            >
              <p className="text-sm text-gray-500">{item.label}</p>
              <p className="mt-2 text-xl font-semibold text-[#2f5d50]">
                {item.value}
              </p>
            </div>
          ))}
        </section>

        {/* ===== BREED KNOWLEDGE BASE ===== */}
        <section className="space-y-12">
          <div className="text-center space-y-2">
            <h2 className="text-3xl font-semibold text-gray-800">
              Breed Knowledge Base
            </h2>
            <p className="text-gray-600">
              Animal categories included in the training dataset
            </p>
          </div>

          {/* CATTLE */}
          <div className="space-y-6">
            <h3 className="text-2xl font-medium text-[#2f5d50]">
              Cattle Breeds
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {cattleBreeds.map((breed) => (
                <div
                  key={breed.name}
                  className="bg-white rounded-lg border-l-4 border-[#6fa98c] p-5 shadow-sm"
                >
                  <h4 className="font-semibold text-gray-800 mb-2">
                    {breed.name}
                  </h4>
                  <p className="text-sm text-gray-600">
                    {breed.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* BUFFALO */}
          <div className="space-y-6">
            <h3 className="text-2xl font-medium text-[#2f5d50]">
              Buffalo Breeds
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {buffaloBreeds.map((breed) => (
                <div
                  key={breed.name}
                  className="bg-white rounded-lg border-l-4 border-[#b08968] p-5 shadow-sm"
                >
                  <h4 className="font-semibold text-gray-800 mb-2">
                    {breed.name}
                  </h4>
                  <p className="text-sm text-gray-600">
                    {breed.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ===== CLASSIFICATION WORKSPACE ===== */}
        <section className="bg-white rounded-2xl shadow-md p-10 space-y-8">
          <div className="text-center">
            <h2 className="text-3xl font-semibold text-gray-800">
              Classification Workspace
            </h2>
            <p className="text-gray-600 mt-2">
              Upload an animal image to initiate AI-based classification
            </p>
          </div>

          <div className="flex flex-col lg:flex-row gap-12 justify-center items-start">
            <ImageUpload />

            <div className="flex-1 bg-[#f9faf8] border border-dashed rounded-xl p-8 text-center">
              <p className="text-gray-500 mb-2">Prediction Output</p>
              <p className="text-lg font-medium text-gray-800">
                Breed: —
              </p>
              <p className="text-gray-600 mt-1">
                Confidence Score: —
              </p>
              <p className="text-sm text-gray-400 mt-6">
                AI model integration pending
              </p>
            </div>
          </div>
        </section>

      </main>

      {/* ================= FOOTER ================= */}
      <footer className="bg-[#2f5d50] text-green-100 text-center py-6 text-sm">
        Image-based Animal Type Classification System • Capstone Project • Phase II
      </footer>

    </div>
  )
}

export default Dashboard
