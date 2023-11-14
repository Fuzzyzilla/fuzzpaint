mod riff;

/// From the given document state reader and repository handle, write a `.fzp` document into the given writer.
pub fn write_into<Document, Writer>(
    document: Document,
    point_repository: &crate::repositories::points::PointRepository,
    writer: Writer,
) -> anyhow::Result<()>
where
    Document: crate::commands::queue::state_reader::CommandQueueStateReader,
    Writer: std::io::Write + std::io::Seek,
{
    todo!()
}
