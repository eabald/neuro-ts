export interface ExampleType {
    id: number;
    name: string;
    isActive: boolean;
}

export type ExampleResponse = {
    data: ExampleType[];
    total: number;
};