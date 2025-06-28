
"use client";

import type React from 'react';
import { useState, useEffect, useRef, useCallback } from 'react';
import {
  SidebarProvider,
  Sidebar,
  SidebarHeader,
  SidebarContent,
  SidebarFooter,
  SidebarInset,
} from '@/components/ui/sidebar';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from '@/components/ui/sheet';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/hooks/use-toast';
import { Bot, User, Send, Loader2, FileText, Paperclip, GripVertical, List, Image as ImageIcon, File as FileIcon, TableIcon as CSVIcon, CodeIcon } from 'lucide-react';
import { codeAssistant } from '@/ai/flows/code-assistant';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
}

interface UploadedFile {
  id: string;
  name: string;
  type: 'image' | 'pdf' | 'csv' | 'code' | 'other';
  content: string; 
  dataUri?: string; 
  mimeType: string;
}

interface StoredUploadedFileMetadata {
  id: string;
  name: string;
  type: UploadedFile['type'];
  mimeType: string;
}


const MIN_SIDEBAR_WIDTH = 180;
const MAX_SIDEBAR_WIDTH = 800;
const SIDEBAR_ICON_WIDTH = 48;
const COLLAPSE_THRESHOLD = 100;
const CSV_PREVIEW_MAX_ROWS = 100;

const CodeBlockDisplay: React.FC<{ language?: string; code: string }> = ({ language, code }) => {
  return (
    <div className="my-2 p-3 bg-background border border-border rounded-md overflow-x-auto">
      {language && (
        <span className="block text-xs text-muted-foreground mb-1">{language}</span>
      )}
      <pre className="whitespace-pre font-code text-sm">
        <code>{code}</code>
      </pre>
    </div>
  );
};

const renderMessageContent = (content: string) => {
  const parts = [];
  let lastIndex = 0;
  const codeBlockRegex = /```(\w*)\n([\s\S]+?)\n```/g;
  let match;

  while ((match = codeBlockRegex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`text-${lastIndex}`}>{content.substring(lastIndex, match.index)}</span>);
    }
    parts.push(
      <CodeBlockDisplay
        key={`code-${match.index}`}
        language={match[1]}
        code={match[2].trim()}
      />
    );
    lastIndex = match.index + match[0].length;
  }

  if (lastIndex < content.length) {
    parts.push(<span key={`text-${lastIndex}`}>{content.substring(lastIndex)}</span>);
  }

  if (parts.length === 0 && content) {
    parts.push(<span key="text-only">{content}</span>);
  }

  return parts;
};

const getFileType = (fileName: string, fileMimeType: string): UploadedFile['type'] => {
  const extension = fileName.split('.').pop()?.toLowerCase();
  if (fileMimeType.startsWith('image/')) return 'image';
  if (extension === 'pdf' || fileMimeType === 'application/pdf') return 'pdf';
  if (extension === 'csv' || fileMimeType === 'text/csv') return 'csv';
  if (['js', 'ts', 'tsx', 'jsx', 'py', 'java', 'c', 'cpp', 'cs', 'html', 'css', 'json', 'md', 'txt'].includes(extension || '')) return 'code';
  return 'other';
};

const parseCSV = (csvText: string, maxRows?: number): { headers: string[]; rows: string[][]; isTruncated: boolean } => {
  try {
    const lines = csvText.trim().split(/\r?\n/);
    if (lines.length === 0) return { headers: [], rows: [], isTruncated: false };
    
    const headers = lines[0].split(',').map(h => h.trim());
    let dataLines = lines.slice(1);
    let isTruncated = false;

    if (maxRows !== undefined && dataLines.length > maxRows) {
      dataLines = dataLines.slice(0, maxRows);
      isTruncated = true;
    }

    const rows = dataLines.map(line => {
      // More robust CSV parsing to handle commas within quoted fields
      const rowValues: string[] = [];
      let currentVal = '';
      let inQuotes = false;
      for (let i = 0; i < line.length; i++) {
        const char = line[i];
        if (char === '"') {
          inQuotes = !inQuotes;
        } else if (char === ',' && !inQuotes) {
          rowValues.push(currentVal.trim());
          currentVal = '';
        } else {
          currentVal += char;
        }
      }
      rowValues.push(currentVal.trim()); // Add the last value
      return rowValues.length === headers.length ? rowValues : headers.map(() => ''); // Ensure row has same number of columns as headers
    });
    
    return { headers, rows, isTruncated };
  } catch (error) {
    console.error("Failed to parse CSV:", error);
    return { headers: ["Error parsing CSV"], rows: [[]], isTruncated: false };
  }
};


export default function FirecodeStudioPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [userInput, setUserInput] = useState('');
  const [isLoadingAI, setIsLoadingAI] = useState(false);

  const [uploadedFiles, setUploadedFiles] = useState<UploadedFile[]>([]);
  const [activeFileId, setActiveFileId] = useState<string | null>(null);
  const [isFileListOpen, setIsFileListOpen] = useState(false);

  const [sidebarWidth, setSidebarWidth] = useState(300);
  const [isSidebarEffectivelyOpen, setIsSidebarEffectivelyOpen] = useState(true);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const chatScrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  const isDraggingRef = useRef(false);
  const startXRef = useRef(0);
  const startWidthRef = useRef(0);
  const pageWrapperRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (sidebarWidth <= COLLAPSE_THRESHOLD) {
      setIsSidebarEffectivelyOpen(false);
    } else {
      setIsSidebarEffectivelyOpen(true);
    }
  }, [sidebarWidth]);

  useEffect(() => {
    try {
      const storedMessages = localStorage.getItem('firecode_chat_messages');
      if (storedMessages) setMessages(JSON.parse(storedMessages));

      const storedUploadedFilesMeta = localStorage.getItem('firecode_uploaded_files_metadata');
      if (storedUploadedFilesMeta) {
        const filesMetadata: StoredUploadedFileMetadata[] = JSON.parse(storedUploadedFilesMeta);
        setUploadedFiles(filesMetadata.map(meta => ({
          ...meta,
          content: '', 
          dataUri: undefined, 
        })));
      }
      
      const storedActiveFileId = localStorage.getItem('firecode_active_file_id');
      if (storedActiveFileId) setActiveFileId(storedActiveFileId);

      const storedSidebarWidth = localStorage.getItem('firecode_sidebar_width');
      if (storedSidebarWidth) setSidebarWidth(parseInt(storedSidebarWidth, 10));

    } catch (error) {
      console.error("Failed to load from localStorage", error);
      toast({
        title: "Error",
        description: "Could not load previous session data.",
        variant: "destructive",
      });
    }
  }, [toast]);

  useEffect(() => {
    try {
      if (messages.length > 0) {
         localStorage.setItem('firecode_chat_messages', JSON.stringify(messages));
      } else if (localStorage.getItem('firecode_chat_messages')) {
         localStorage.removeItem('firecode_chat_messages');
      }
    } catch (error) {
      console.error("Failed to save messages to localStorage", error);
    }
  }, [messages]);

  useEffect(() => {
    try {
      const filesToStore: StoredUploadedFileMetadata[] = uploadedFiles.map(file => ({
        id: file.id,
        name: file.name,
        type: file.type,
        mimeType: file.mimeType,
      }));
      if (filesToStore.length > 0) {
        localStorage.setItem('firecode_uploaded_files_metadata', JSON.stringify(filesToStore));
      } else if (localStorage.getItem('firecode_uploaded_files_metadata')) {
        localStorage.removeItem('firecode_uploaded_files_metadata');
      }
    } catch (error) {
      console.error("Failed to save uploaded files metadata to localStorage", error);
       toast({
        title: "Storage Warning",
        description: "Could not save full file list state. Some file info might be lost on refresh.",
        variant: "default", 
      });
    }
  }, [uploadedFiles, toast]);
  
  useEffect(() => {
    try {
      if (activeFileId) {
        localStorage.setItem('firecode_active_file_id', activeFileId);
      } else {
        localStorage.removeItem('firecode_active_file_id');
      }
    } catch (error) {
      console.error("Failed to save active file ID to localStorage", error);
    }
  }, [activeFileId]);


  useEffect(() => {
    try {
      localStorage.setItem('firecode_sidebar_width', sidebarWidth.toString());
    } catch (error) {
      console.error("Failed to save sidebar width to localStorage", error);
    }
  }, [sidebarWidth]);


  useEffect(() => {
    if (chatScrollAreaRef.current) {
      const scrollViewport = chatScrollAreaRef.current.querySelector('div[data-radix-scroll-area-viewport]');
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [messages]);

  const handleMouseMove = useCallback((event: MouseEvent) => {
    if (!isDraggingRef.current || !pageWrapperRef.current) return;
    event.preventDefault();

    const currentX = event.clientX;
    const deltaX = currentX - startXRef.current;
    let newWidth = startWidthRef.current + deltaX;

    if (newWidth < SIDEBAR_ICON_WIDTH + 20 && newWidth > SIDEBAR_ICON_WIDTH - 20) {
        newWidth = SIDEBAR_ICON_WIDTH;
    } else {
        newWidth = Math.max(MIN_SIDEBAR_WIDTH, Math.min(newWidth, MAX_SIDEBAR_WIDTH));
    }

    const screenMaxW = window.innerWidth * 0.7;
    newWidth = Math.min(newWidth, screenMaxW);

    setSidebarWidth(newWidth);
  }, []);

  const handleMouseUp = useCallback(() => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    if (pageWrapperRef.current) {
        pageWrapperRef.current.classList.remove('cursor-ew-resize');
        pageWrapperRef.current.style.userSelect = '';
    }
  }, [handleMouseMove]);

  const handleMouseDown = useCallback((event: React.MouseEvent<HTMLDivElement>) => {
    isDraggingRef.current = true;
    startXRef.current = event.clientX;
    startWidthRef.current = sidebarWidth;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    event.preventDefault();
    if (pageWrapperRef.current) {
        pageWrapperRef.current.classList.add('cursor-ew-resize');
        pageWrapperRef.current.style.userSelect = 'none';
    }
  }, [sidebarWidth, handleMouseMove, handleMouseUp]);

  useEffect(() => {
    return () => {
      if (isDraggingRef.current) {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
         if (pageWrapperRef.current) {
            pageWrapperRef.current.classList.remove('cursor-ew-resize');
            pageWrapperRef.current.style.userSelect = '';
        }
      }
    };
  }, [handleMouseMove, handleMouseUp]);

  const getCurrentCodeForAI = useCallback(() => {
    if (!activeFileId) return '';
    const activeFile = uploadedFiles.find(f => f.id === activeFileId);
    if (activeFile && (activeFile.type === 'csv' || activeFile.type === 'code' || activeFile.type === 'other' || activeFile.type === 'pdf')) {
      return activeFile.content || ''; 
    }
    return '';
  }, [activeFileId, uploadedFiles]);

  const handleSendMessage = useCallback(async () => {
    if (!userInput.trim()) return;

    const newUserMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: userInput.trim(),
    };
    setMessages((prev) => [...prev, newUserMessage]);
    setUserInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
      if (textareaRef.current.scrollHeight <= 40 ) {
        textareaRef.current.style.height = '40px';
      }
       textareaRef.current.style.overflowY = 'hidden';
    }

    setIsLoadingAI(true);

    try {
      const response = await codeAssistant({
        code: getCurrentCodeForAI(),
        question: newUserMessage.content,
      });
      const aiMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
      };
      setMessages((prev) => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error calling AI assistant:', error);
      toast({
        title: 'AI Error',
        description: 'Failed to get response from AI assistant.',
        variant: 'destructive',
      });
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'system',
        content: 'Error: Could not connect to the AI assistant.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoadingAI(false);
    }
  }, [userInput, getCurrentCodeForAI, toast]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      const fileId = `${file.name}-${Date.now()}`;
      const fileType = getFileType(file.name, file.type);

      reader.onload = async (e) => {
        const result = e.target?.result as string;
        let newFile: UploadedFile;

        if (fileType === 'image') {
          newFile = { id: fileId, name: file.name, type: fileType, dataUri: result, content: '', mimeType: file.type };
        } else {
          newFile = { id: fileId, name: file.name, type: fileType, content: result, mimeType: file.type };
        }

        setUploadedFiles(prev => [...prev, newFile]);
        setActiveFileId(fileId);

        const systemMessage: Message = {
          id: Date.now().toString(),
          role: 'system',
          content: `File "${file.name}" loaded. It's now the active file for review.`,
        };
        setMessages((prev) => [...prev, systemMessage]);
        toast({
          title: 'File Loaded',
          description: `${file.name} is now active.`,
        });
      };
      reader.onerror = () => {
        toast({
          title: 'File Error',
          description: `Could not read file ${file.name}.`,
          variant: 'destructive',
        });
      };

      if (fileType === 'image') {
        reader.readAsDataURL(file);
      } else {
        reader.readAsText(file);
      }
    }
    if (fileInputRef.current) {
        fileInputRef.current.value = "";
    }
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleInputChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setUserInput(event.target.value);
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      const scrollHeight = textarea.scrollHeight;
      const maxHeight = 150; 

      if (scrollHeight > maxHeight) {
        textarea.style.height = `${maxHeight}px`;
        textarea.style.overflowY = 'auto';
      } else {
        textarea.style.height = `${scrollHeight}px`;
        textarea.style.overflowY = 'hidden';
      }
    }
  };

  const renderActiveFileContent = () => {
    if (!activeFileId) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
          <FileText size={64} className="mb-4" />
          <p className="text-lg font-medium">No file selected for review</p>
          <p className="text-sm">Upload a file or select one from the list.</p>
        </div>
      );
    }
    const fileToRender = uploadedFiles.find(f => f.id === activeFileId);
    if (!fileToRender) {
      return (
        <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
          <FileText size={64} className="mb-4" />
          <p className="text-lg font-medium">File not found</p>
        </div>
      );
    }
    
    if (fileToRender.type !== 'image' && !fileToRender.content) {
        return (
             <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-4">
                <FileText size={64} className="mb-4" />
                <p className="text-lg font-medium">Content for {fileToRender.name} is not available.</p>
                <p className="text-sm">File content is not stored between sessions to save space. Please re-upload if needed.</p>
            </div>
        );
    }
     if (fileToRender.type === 'image' && !fileToRender.dataUri) {
        return (
             <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-4">
                <ImageIcon size={64} className="mb-4" />
                <p className="text-lg font-medium">Image data for {fileToRender.name} is not available.</p>
                <p className="text-sm">Image content is not stored between sessions to save space. Please re-upload if needed.</p>
            </div>
        );
    }


    switch (fileToRender.type) {
      case 'image':
        return <img src={fileToRender.dataUri} alt={fileToRender.name} className="max-w-full max-h-full object-contain p-2" data-ai-hint="uploaded image" />;
      case 'csv':
        const { headers, rows, isTruncated } = parseCSV(fileToRender.content, CSV_PREVIEW_MAX_ROWS);
        if (headers.length === 1 && headers[0] === "Error parsing CSV") {
            return <p className="p-4 text-destructive-foreground bg-destructive rounded-md">Error displaying CSV: Could not parse the file.</p>;
        }
        return (
          <ScrollArea className="whitespace-normal rounded-md border p-2 bg-card shadow w-96">
            {isTruncated && (
              <p className="text-sm text-muted-foreground mb-2 p-2 whitespace-normal">
                Showing a preview of the first {CSV_PREVIEW_MAX_ROWS} data rows of "{fileToRender.name}". Full content is available to the AI.
              </p>
            )}
            <Table>
              <TableHeader>
                <TableRow>
                  {headers.map((header, idx) => <TableHead key={`${header}-${idx}`}>{header}</TableHead>)}
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((row, rowIndex) => (
                  <TableRow key={`row-${rowIndex}`}>
                    {row.map((cell, cellIndex) => <TableCell key={`cell-${rowIndex}-${cellIndex}`}>{cell}</TableCell>)}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        );
      case 'pdf':
        return (
           <div className="p-4">
            <h3 className="font-semibold text-lg mb-2">PDF File: {fileToRender.name}</h3>
            <p className="text-muted-foreground mb-2">Direct PDF preview is not available. Showing first 500 characters of text content if available:</p>
            <pre className="text-sm font-code whitespace-pre-wrap break-all p-2 bg-card rounded-md shadow max-h-96 overflow-y-auto">
              <code>{fileToRender.content ? fileToRender.content.substring(0, 500) + (fileToRender.content.length > 500 ? '...' : ''): "No text content extracted or PDF is binary."}</code>
            </pre>
          </div>
        );
      case 'code':
      case 'other':
        return (
          <pre className="text-sm font-code whitespace-pre-wrap break-all p-2 bg-card rounded-md shadow overflow-x-auto">
            <code>{fileToRender.content}</code>
          </pre>
        );
      default:
        return <p>Unsupported file type: {fileToRender.name}</p>;
    }
  };
  
  const getFileIcon = (fileType: UploadedFile['type']) => {
    switch (fileType) {
      case 'image': return <ImageIcon className="h-5 w-5 mr-2 text-blue-400" />;
      case 'pdf': return <FileIcon className="h-5 w-5 mr-2 text-red-400" />;
      case 'csv': return <CSVIcon className="h-5 w-5 mr-2 text-green-400" />;
      case 'code': return <CodeIcon className="h-5 w-5 mr-2 text-purple-400" />;
      default: return <FileText className="h-5 w-5 mr-2 text-gray-400" />;
    }
  };


  return (
    <SidebarProvider
      open={isSidebarEffectivelyOpen}
      onOpenChange={setIsSidebarEffectivelyOpen}
      style={{
        "--sidebar-width": `${sidebarWidth}px`,
        "--sidebar-width-icon": `${SIDEBAR_ICON_WIDTH}px`
      } as React.CSSProperties}
    >
      <div className="flex h-screen w-screen bg-background text-foreground overflow-hidden" ref={pageWrapperRef}>
        <Sidebar
          side="left"
          collapsible="icon"
        >
          <SidebarHeader className="p-4 border-b border-sidebar-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Bot size={28} className="text-primary" />
                <h1 className="text-xl font-headline font-semibold group-data-[state=collapsed]:hidden">
                  Firecode Studio
                </h1>
              </div>
            </div>
          </SidebarHeader>

          <SidebarContent className="p-0 group-data-[state=collapsed]:p-2">
            <ScrollArea className="h-full group-data-[state=expanded]:p-4" ref={chatScrollAreaRef}>
              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex items-start gap-3 mb-4 p-3 rounded-lg shadow-sm ${
                    msg.role === 'user' ? 'ml-auto bg-secondary max-w-[85%]' :
                    msg.role === 'assistant' ? 'bg-card max-w-[85%]' :
                    'bg-muted text-muted-foreground text-xs italic max-w-full text-center'
                  } group-data-[state=collapsed]:hidden`}
                >
                  {msg.role !== 'system' && (
                    <Avatar className="w-8 h-8 shrink-0">
                      <AvatarFallback className={msg.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-accent text-accent-foreground'}>
                        {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
                      </AvatarFallback>
                    </Avatar>
                  )}
                  <div className={`whitespace-pre-wrap break-words ${msg.role === 'system' ? 'w-full' : ''}`}>
                    {renderMessageContent(msg.content)}
                  </div>
                </div>
              ))}
               {messages.length === 0 && !isLoadingAI && (
                <div className="text-center text-muted-foreground p-8 group-data-[state=collapsed]:hidden">
                  <FileText size={48} className="mx-auto mb-2" />
                  <p className="font-medium">Welcome to Firecode Studio!</p>
                  <p className="text-sm">Import a file or ask a question to get started.</p>
                </div>
              )}
              {isLoadingAI && (
                <div className="flex items-center gap-3 mb-4 p-3 rounded-lg shadow-sm bg-card max-w-[85%] group-data-[state=collapsed]:hidden">
                   <Avatar className="w-8 h-8 shrink-0">
                      <AvatarFallback className='bg-accent text-accent-foreground'>
                        <Bot size={18} />
                      </AvatarFallback>
                    </Avatar>
                  <Loader2 className="animate-spin text-primary" size={20} />
                  <span className="text-muted-foreground">Assistant is typing...</span>
                </div>
              )}
            </ScrollArea>

             {isSidebarEffectivelyOpen === false && sidebarWidth > SIDEBAR_ICON_WIDTH && (
                 <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                    <Bot size={32} />
                 </div>
             )}
             {sidebarWidth <= SIDEBAR_ICON_WIDTH && (
                <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-2">
                    <Bot size={24} className="text-primary"/>
                 </div>
             )}
          </SidebarContent>

          <SidebarFooter className="p-4 border-t border-sidebar-border group-data-[state=collapsed]:hidden">
            <div className="rounded-lg p-0.5 bg-gradient-to-r from-pink-500 via-orange-500 to-yellow-400">
              <div className="flex items-end gap-2 bg-card rounded-[7px] p-2">
                <Textarea
                  ref={textareaRef}
                  value={userInput}
                  onChange={handleInputChange}
                  placeholder="Describe the changes you want to make"
                  className="flex-grow resize-none min-h-[40px] bg-transparent placeholder-muted-foreground focus-visible:ring-0 border-0 focus-visible:ring-offset-0 appearance-none overflow-hidden"
                  rows={1}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  disabled={isLoadingAI}
                />
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={triggerFileInput}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label="Attach file"
                  disabled={isLoadingAI}
                >
                  <Paperclip size={20} />
                </Button>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileUpload}
                  accept=".png,.jpg,.jpeg,.gif,.pdf,.csv,.txt,.js,.ts,.tsx,.jsx,.py,.java,.c,.cpp,.cs,.html,.css,.json,.md,.*"
                  className="hidden"
                />
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleSendMessage}
                  disabled={isLoadingAI || !userInput.trim()}
                  className="text-muted-foreground hover:text-foreground"
                  aria-label="Send message"
                >
                  {isLoadingAI ? (
                    <Loader2 className="animate-spin" size={20}/>
                  ) : (
                    <Send size={20} />
                  )}
                </Button>
              </div>
            </div>
          </SidebarFooter>
        </Sidebar>

        <div
          onMouseDown={handleMouseDown}
          className="w-2.5 h-full cursor-ew-resize bg-border hover:bg-primary/30 transition-colors flex-shrink-0 flex items-center justify-center group"
        >
          <GripVertical size={16} className="text-muted-foreground group-hover:text-foreground transition-colors" />
        </div>

        <SidebarInset className="flex-1 flex flex-col overflow-hidden">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-lg font-headline font-semibold">Review Panel</h2>
            <Sheet open={isFileListOpen} onOpenChange={setIsFileListOpen}>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon" aria-label="Uploaded files">
                  <List size={20} />
                </Button>
              </SheetTrigger>
              <SheetContent side="right" className="w-[350px] sm:w-[450px] p-0">
                <SheetHeader className="p-4 border-b">
                  <SheetTitle>Uploaded Files</SheetTitle>
                </SheetHeader>
                <ScrollArea className="h-[calc(100%-57px)]">
                  {uploadedFiles.length === 0 ? (
                    <p className="p-4 text-sm text-muted-foreground">No files uploaded yet.</p>
                  ) : (
                    <div className="p-2">
                    {uploadedFiles.map(file => (
                      <Button
                        key={file.id}
                        variant={activeFileId === file.id ? "secondary" : "ghost"}
                        className="w-full justify-start mb-1 h-auto py-2 px-3 text-left"
                        onClick={() => {
                          setActiveFileId(file.id);
                          setIsFileListOpen(false);
                        }}
                      >
                        <div className="flex items-center w-full">
                           {getFileIcon(file.type)}
                           <div className="flex flex-col ml-1 overflow-hidden">
                             <span className="font-medium truncate">{file.name}</span>
                             <span className="text-xs text-muted-foreground">
                               {file.type.toUpperCase()} - 
                               {(file.content || file.dataUri) ? 
                                 `${((file.content?.length || file.dataUri?.length || 0) / 1024).toFixed(2)} KB` : 
                                 'N/A (metadata only)'}
                             </span>
                           </div>
                        </div>
                      </Button>
                    ))}
                    </div>
                  )}
                </ScrollArea>
              </SheetContent>
            </Sheet>
          </div>
          <ScrollArea className="flex-1 p-4 bg-muted/30">
            {renderActiveFileContent()}
          </ScrollArea>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
}

