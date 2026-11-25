import { useEffect, useRef, useState } from "react";

interface Props {
  file: File | null;
  onFile: (file: File) => void;
  onClear: () => void;
}

export function ImageDrop({ file, onFile, onClear }: Props) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const pick = (files: FileList | null) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <div
      className={`drop ${dragging ? "drop--active" : ""} ${preview ? "drop--filled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        pick(e.dataTransfer.files);
      }}
    >
      {preview ? (
        <>
          <img src={preview} alt="Selected product" className="drop__img" />
          <div className="drop__bar">
            <span className="drop__name">{file?.name}</span>
            <button type="button" className="link" onClick={onClear}>
              Remove
            </button>
          </div>
        </>
      ) : (
        <button type="button" className="drop__empty" onClick={() => input.current?.click()}>
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor"
               strokeWidth="1.4" aria-hidden="true">
            <path d="M8 4 4 6.5 5.5 10 7 9.3V20h10V9.3l1.5.7L20 6.5 16 4c-.6 1.4-2.1 2.3-4 2.3S8.6 5.4 8 4Z" />
          </svg>
          <span className="drop__title">Drop a product photo here</span>
          <span className="drop__hint">or click to choose a file · JPG or PNG, up to 10 MB</span>
        </button>
      )}
      <input
        ref={input}
        type="file"
        accept="image/*"
        hidden
        data-testid="file-input"
        onChange={(e) => {
          pick(e.target.files);
          e.target.value = "";
        }}
      />
    </div>
  );
}
