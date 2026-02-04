const ImagePreview = ({ image }) => {
  return (
    <div className="mt-6">
      
      <p className="text-sm text-gray-600 mb-2">
        Selected Image Preview
      </p>

      <div className="border rounded-lg p-3">
        <img
          src={URL.createObjectURL(image)}
          alt="preview"
          className="w-full h-56 object-contain rounded"
        />
      </div>

      <div className="mt-3 text-sm text-gray-700">
        <p><strong>File Name:</strong> {image.name}</p>
        <p><strong>File Size:</strong> {(image.size / 1024).toFixed(2)} KB</p>
      </div>

    </div>
  )
}

export default ImagePreview
