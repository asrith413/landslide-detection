use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: parquet_extract <file>");
    let file = File::open(&path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    println!("SCHEMA:\n{:#?}", builder.schema());
    let meta = builder.metadata();
    println!("rows={} row_groups={}", meta.file_metadata().num_rows(), meta.num_row_groups());
    Ok(())
}
