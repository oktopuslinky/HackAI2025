import { useState, useEffect } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import axios from "axios"

export default function ChatbotPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [uploadedPdf, setUploadedPdf] = useState(null)

  useEffect(() => {
    const filename = localStorage.getItem('uploadedPdfFilename')
    setUploadedPdf(filename)
  }, [])

  const handleSend = async () => {
    if (!input.trim()) return
    const userMessage = { role: "user", content: input }
    setMessages(prev => [...prev, userMessage])

    try {
      const response = await axios.post("http://127.0.0.1:8000/query", { query: input })
      const botMessage = { role: "bot", content: response.data.answer }
      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error(error)
    }

    setInput("")
  }

  return (
    <div className="flex min-h-screen bg-black text-white">
      
      {/* Left side: Chat area */}
      <div className="w-1/2 p-6 flex flex-col space-y-4 overflow-y-auto">
        <h1 className="text-3xl font-bold mb-4">Chat about your document</h1>

        <div className="flex-1 space-y-4">
          {messages.map((msg, index) => (
            <div key={index} className={`p-4 rounded-xl ${msg.role === "user" ? "bg-purple-700" : "bg-gray-800"}`}>
              {msg.content}
            </div>
          ))}
        </div>

        <div className="flex space-x-2 mt-6">
          <Input
            className="flex-1"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something about your PDF..."
          />
          <Button onClick={handleSend}>Send</Button>
        </div>
      </div>

      {/* Right side: PDF viewer */}
      <div className="w-1/2 p-4 bg-[#111111] flex justify-center items-center">
        {uploadedPdf ? (
          <iframe
            src={`http://127.0.0.1:8000/static/${uploadedPdf}`}
            className="w-full h-full rounded-2xl"
            title="Uploaded PDF"
            /> 
        ) : (
          <p>Loading PDF...</p>
        )}
      </div>

    </div>
  )
}
