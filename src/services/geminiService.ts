import { GoogleGenAI } from '@google/genai';

// Note: In production, this should be handled via a backend proxy
const API_KEY = (import.meta.env.VITE_GEMINI_API_KEY || 'demo-key') as string;
const genAI = new GoogleGenAI({ apiKey: API_KEY });

export interface GenerationRequest {
  prompt: string;
  referenceImages?: string[]; // base64 array
  temperature?: number;
  seed?: number;
}

export interface EditRequest {
  instruction: string;
  originalImage: string; // base64
  referenceImages?: string[]; // base64 array
  maskImage?: string; // base64
  temperature?: number;
  seed?: number;
}

export interface SegmentationRequest {
  image: string; // base64
  query: string; // "the object at pixel (x,y)" or "the red car"
}

export class GeminiService {
  async generateImage(request: GenerationRequest): Promise<string[]> {
    try {
      const contents: any[] = [{ text: request.prompt }];

      // Add reference images if provided
      if (request.referenceImages && request.referenceImages.length > 0) {
        request.referenceImages.forEach(image => {
          contents.push({
            inlineData: {
              mimeType: "image/png",
              data: image,
            },
          });
        });
      }

      const response = await genAI.models.generateContent({
        model: "gemini-2.5-flash-image-preview",
        contents,
      });

      const images: string[] = [];

      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData?.data) {
            images.push(part.inlineData.data);
          }
        }
      }

      return images;
    } catch (error: any) {
      console.error('Error generating image:', error);

      // Handle rate limit errors specifically (multiple possible error structures)
      if (error?.error?.code === 429 ||
        error?.error?.error?.code === 429 ||
        error?.code === 429 ||
        (error?.error?.message && error.error.message.includes('quota'))) {
        throw new Error('Rate limit exceeded. Please wait a few minutes before trying again, or upgrade your Google AI Studio plan for higher quotas.');
      }

      // Handle other API errors
      if (error?.error?.message) {
        throw new Error(`API Error: ${error.error.message}`);
      }

      if (error?.error?.error?.message) {
        throw new Error(`API Error: ${error.error.error.message}`);
      }

      if (error?.message) {
        throw new Error(`Error: ${error.message}`);
      }

      throw new Error('Failed to generate image. Please try again.');
    }
  }

  async editImage(request: EditRequest): Promise<string[]> {
    try {
      const contents = [
        { text: this.buildEditPrompt(request) },
        {
          inlineData: {
            mimeType: "image/png",
            data: request.originalImage,
          },
        },
      ];

      // Add reference images if provided
      if (request.referenceImages && request.referenceImages.length > 0) {
        request.referenceImages.forEach(image => {
          contents.push({
            inlineData: {
              mimeType: "image/png",
              data: image,
            },
          });
        });
      }

      if (request.maskImage) {
        contents.push({
          inlineData: {
            mimeType: "image/png",
            data: request.maskImage,
          },
        });
      }

      const response = await genAI.models.generateContent({
        model: "gemini-2.5-flash-image-preview",
        contents,
      });

      const images: string[] = [];

      if (response.candidates?.[0]?.content?.parts) {
        for (const part of response.candidates[0].content.parts) {
          if (part.inlineData?.data) {
            images.push(part.inlineData.data);
          }
        }
      }

      return images;
    } catch (error) {
      console.error('Error editing image:', error);
      throw new Error('Failed to edit image. Please try again.');
    }
  }

  async segmentImage(request: SegmentationRequest): Promise<any> {
    try {
      const prompt = [
        {
          text: `Analyze this image and create a segmentation mask for: ${request.query}

Return a JSON object with this exact structure:
{
  "masks": [
    {
      "label": "description of the segmented object",
      "box_2d": [x, y, width, height],
      "mask": "base64-encoded binary mask image"
    }
  ]
}

Only segment the specific object or region requested. The mask should be a binary PNG where white pixels (255) indicate the selected region and black pixels (0) indicate the background.` },
        {
          inlineData: {
            mimeType: "image/png",
            data: request.image,
          },
        },
      ];

      const response = await genAI.models.generateContent({
        model: "gemini-2.5-flash-image-preview",
        contents: prompt,
      });

      const responseText = response.candidates?.[0]?.content?.parts?.[0]?.text;
      if (!responseText) {
        throw new Error('No response text received from API');
      }
      return JSON.parse(responseText);
    } catch (error) {
      console.error('Error segmenting image:', error);
      throw new Error('Failed to segment image. Please try again.');
    }
  }

  private buildEditPrompt(request: EditRequest): string {
    const maskInstruction = request.maskImage
      ? "\n\nIMPORTANT: Apply changes ONLY where the mask image shows white pixels (value 255). Leave all other areas completely unchanged. Respect the mask boundaries precisely and maintain seamless blending at the edges."
      : "";

    return `Edit this image according to the following instruction: ${request.instruction}

Maintain the original image's lighting, perspective, and overall composition. Make the changes look natural and seamlessly integrated.${maskInstruction}

Preserve image quality and ensure the edit looks professional and realistic.`;
  }
}

export const geminiService = new GeminiService();