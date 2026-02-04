import ImageUpload from "../components/ImageUpload"

const Dashboard = () => {
  return (
    <div className="max-w-5xl mx-auto px-4 py-10">
      
      <h1 className="text-3xl font-bold text-gray-800 text-center">
        Image-Based Animal Type Classification
      </h1>

      <p className="text-center text-gray-600 mt-3">
        Upload an image of cattle or buffalo to initiate AI-based classification
      </p>

      <div className="mt-10 flex justify-center">
        <ImageUpload />
      </div>

    </div>
  )
}

export default Dashboard
