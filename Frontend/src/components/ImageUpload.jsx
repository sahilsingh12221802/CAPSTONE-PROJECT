import { useState } from "react"
import ImagePreview from "./ImagePreview"

const ImageUpload = () => {
  const [image, setImage] = useState(null)

  const handleImageChange = (e) => {
    const file = e.target.files[0]
    if (file) setImage(file)
  }

  return (
    <div className="bg-white shadow-lg rounded-xl p-6 w-full max-w-md">

      <h2 className="text-xl font-semibold text-gray-700 mb-4">
        Upload Animal Image
      </h2>

      <label className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-6 cursor-pointer hover:border-blue-500 transition">
        
        <span className="text-gray-500 text-sm">
          Click to upload JPG or PNG image
        </span>

        <input
          type="file"
          accept="image/png, image/jpeg"
          className="hidden"
          onChange={handleImageChange}
        />
      </label>

      {image && <ImagePreview image={image} />}

    </div>
  )
}

export default ImageUpload
