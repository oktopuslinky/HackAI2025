import { useEffect, useState } from "react"

export default function PdfViewer({ file }) {
  const [pdfUrl, setPdfUrl] = useState(null)

  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file)
      setPdfUrl(objectUrl)

      return () => URL.revokeObjectURL(objectUrl)
    }
  }, [file])

  if (!file) return null

  return (
    <div className="flex justify-center mt-20">
      <div className="w-full max-w-[95%] rounded-2xl overflow-hidden bg-[#0d0d0d] shadow-lg border border-gray-800">
        <iframe
          src={pdfUrl}
          title="Uploaded PDF"
          className="w-full h-[90vh] min-h-[700px] rounded-2xl"
        />
      </div>
    </div>
  )
}
