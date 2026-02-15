"use client"

import React, { useState, useRef } from "react"

interface ColorPickerProps {
  tokenKey: string
  value: string
  label: string
  onChange: (key: string, value: string) => void
}

export default function ColorPicker({ tokenKey, value, label, onChange }: ColorPickerProps) {
  const [inputValue, setInputValue] = useState(value)
  const colorInputRef = useRef<HTMLInputElement>(null)

  const handleBlur = () => {
    if (/^#[0-9a-fA-F]{6}$/.test(inputValue)) {
      onChange(tokenKey, inputValue.toUpperCase())
    } else {
      setInputValue(value)
    }
  }

  const handleColorChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const hex = e.target.value.toUpperCase()
    setInputValue(hex)
    onChange(tokenKey, hex)
  }

  React.useEffect(() => {
    setInputValue(value)
  }, [value])

  return (
    <div className="flex items-center gap-3 group">
      <button
        onClick={() => colorInputRef.current?.click()}
        className="relative h-8 w-8 rounded-md border border-border shrink-0 cursor-pointer overflow-hidden"
        style={{ backgroundColor: value }}
      >
        <input
          ref={colorInputRef}
          type="color"
          value={value}
          onChange={handleColorChange}
          className="absolute inset-0 opacity-0 cursor-pointer"
        />
      </button>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-muted-foreground truncate">{label}</p>
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onBlur={handleBlur}
          className="w-full text-xs font-mono text-foreground bg-transparent border-none outline-none p-0"
          spellCheck={false}
        />
      </div>
    </div>
  )
}
