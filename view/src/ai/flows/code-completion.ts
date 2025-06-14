// This file holds the code completion flow.
'use server';
/**
 * @fileOverview Provides code completion suggestions based on the current code and comments.
 *
 * - getCodeCompletionSuggestions - A function that takes code and comments as input and returns code completion suggestions.
 * - CodeCompletionInput - The input type for the getCodeCompletionSuggestions function.
 * - CodeCompletionOutput - The return type for the getCodeCompletionSuggestions function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const CodeCompletionInputSchema = z.object({
  code: z.string().describe('The current code.'),
  comments: z.string().describe('The comments related to the code.'),
});

export type CodeCompletionInput = z.infer<typeof CodeCompletionInputSchema>;

const CodeCompletionOutputSchema = z.object({
  suggestions: z.array(z.string()).describe('An array of code completion suggestions.'),
});

export type CodeCompletionOutput = z.infer<typeof CodeCompletionOutputSchema>;

export async function getCodeCompletionSuggestions(input: CodeCompletionInput): Promise<CodeCompletionOutput> {
  return codeCompletionFlow(input);
}

const prompt = ai.definePrompt({
  name: 'codeCompletionPrompt',
  input: {schema: CodeCompletionInputSchema},
  output: {schema: CodeCompletionOutputSchema},
  prompt: `You are a code completion assistant. Based on the current code and comments, provide code completion suggestions.

Current Code:
{{code}}

Comments:
{{comments}}

Suggestions:
`,
});

const codeCompletionFlow = ai.defineFlow(
  {
    name: 'codeCompletionFlow',
    inputSchema: CodeCompletionInputSchema,
    outputSchema: CodeCompletionOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
