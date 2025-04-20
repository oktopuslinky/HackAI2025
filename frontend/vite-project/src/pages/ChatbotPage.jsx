"use client"

import { useEffect, useRef, useState } from "react"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import axios from "axios"
import { Loader2, Send, FileText, MessageSquare, ChevronLeft } from "lucide-react"
import { Avatar } from "@/components/ui/avatar"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

export default function ChatbotPage() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState("")
  const [uploadedPdf, setUploadedPdf] = useState(null)
  const [isBotTyping, setIsBotTyping] = useState(false)
  const [isFullscreenChat, setIsFullscreenChat] = useState(false)

  const chatEndRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    const filename = localStorage.getItem("uploadedPdfFilename")
    setUploadedPdf(filename)
  }, [])

  useEffect(() => {
    // Always scroll to the bottom when messages change
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  useEffect(() => {
    // Focus the input field when the component mounts
    inputRef.current?.focus()
  }, [])

  const handleSend = async () => {
    if (!input.trim()) return
    const userMessage = { role: "user", content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsBotTyping(true)

    try {
      const response = await axios.post("http://127.0.0.1:8000/query", { query: input })
      const botMessage = { role: "bot", content: response.data.answer }
      setMessages((prev) => [...prev, botMessage])
    } catch (error) {
      console.error(error)
      const errorMsg = { role: "bot", content: "Sorry, something went wrong. Please try again." }
      setMessages((prev) => [...prev, errorMsg])
    } finally {
      setIsBotTyping(false)
      inputRef.current?.focus()
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const toggleFullscreenChat = () => {
    setIsFullscreenChat((prev) => !prev)
  }

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-gray-900 to-black text-white">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-10">
        <div className="container mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="h-5 w-5 text-purple-400" />
            <h1 className="text-xl font-semibold">RAGs2Riches</h1>
          </div>
          <Button variant="ghost" size="sm" className="text-gray-400 hover:text-white" onClick={toggleFullscreenChat}>
            {isFullscreenChat ? (
              <>
                <ChevronLeft className="h-4 w-4 mr-1" /> Show PDF
              </>
            ) : (
              <>Full Chat</>
            )}
          </Button>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Left side: Chat */}
        <div
          className={cn(
            "flex flex-col transition-all duration-300 ease-in-out",
            isFullscreenChat ? "w-full" : "w-1/2 md:w-2/5",
          )}
        >
          {/* Chat messages container */}
          <div className="flex-1 overflow-y-auto p-4 scrollbar-thin scrollbar-thumb-gray-700">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8 text-gray-400">
                <MessageSquare className="h-12 w-12 mb-4 text-purple-500 opacity-80" />
                <h3 className="text-xl font-medium mb-2 text-gray-300">Chat with your PDF</h3>
                <p className="max-w-md">
                  Ask questions about the document and get instant answers. Your conversation will appear here.
                </p>
              </div>
            ) : (
              <div className="flex flex-col space-y-6">
                {messages.map((msg, index) => (
                  <div key={index} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div className="flex items-start gap-3 max-w-[85%]">
                      {msg.role !== "user" && (
                        <Avatar className="h-8 w-8 bg-purple-700 flex items-center justify-center">
                          <FileText className="h-4 w-4" />
                        </Avatar>
                      )}
                      <div
                        className={cn(
                          "p-3 rounded-2xl text-sm",
                          msg.role === "user"
                            ? "bg-gradient-to-br from-purple-600 to-purple-800 text-white rounded-tr-none"
                            : "bg-gray-800 text-gray-100 rounded-tl-none",
                        )}
                      >
                        {msg.content}
                      </div>
                      {msg.role === "user" && (
                        <Avatar className="h-8 w-8 bg-indigo-600 flex items-center justify-center">
                          <span className="text-xs font-medium">You</span>
                        </Avatar>
                      )}
                    </div>
                  </div>
                ))}

                {isBotTyping && (
                  <div className="flex justify-start">
                    <div className="flex items-start gap-3 max-w-[85%]">
                      <Avatar className="h-8 w-8 bg-purple-700 flex items-center justify-center">
                        <FileText className="h-4 w-4" />
                      </Avatar>
                      <div className="p-3 rounded-2xl bg-gray-800 text-gray-100 rounded-tl-none">
                        <div className="flex items-center gap-2">
                          <div className="flex space-x-1">
                            <div className="h-2 w-2 bg-purple-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                            <div className="h-2 w-2 bg-purple-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                            <div className="h-2 w-2 bg-purple-400 rounded-full animate-bounce"></div>
                          </div>
                          <span className="text-sm text-gray-400">Thinking...</span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t border-gray-800 bg-gray-900/80 backdrop-blur-sm p-4">
            <div className="flex items-center gap-2 relative">
              <Input
                ref={inputRef}
                className="flex-1 bg-gray-800 border-gray-700 rounded-full pl-4 pr-12 py-6 text-sm focus-visible:ring-1 focus-visible:ring-purple-500 focus-visible:ring-offset-0"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Ask something about your PDF..."
              />
              <Button
                onClick={handleSend}
                disabled={!input.trim() || isBotTyping}
                className="absolute right-1.5 h-9 w-9 rounded-full bg-gradient-to-r from-purple-600 to-indigo-600 p-0 flex items-center justify-center"
                aria-label="Send message"
              >
                <Send className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Separator */}
        {!isFullscreenChat && <Separator orientation="vertical" className="bg-gray-800" />}

        {/* Right side: PDF Viewer */}
        {!isFullscreenChat && (
  <div className="w-1/2 md:w-3/5 flex bg-black">
    {uploadedPdf ? (
      <iframe
        src={`http://127.0.0.1:8000/static/${uploadedPdf}`}
        className="w-full h-full"
        title="Uploaded PDF"
      />
    ) : (
      <div className="flex-1 flex items-center justify-center">
        <div className="flex flex-col items-center text-gray-400">
          <Loader2 className="h-8 w-8 animate-spin mb-2" />
          <p>Loading PDF...</p>
        </div>
      </div>
    )}
  </div>
)}

      </div>
    </div>
  )
}
